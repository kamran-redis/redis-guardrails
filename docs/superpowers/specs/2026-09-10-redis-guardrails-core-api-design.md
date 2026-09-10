# Redis Guardrails Core API — Design

## Status

Approved design. `docs/vector-guardrail-algorithm.md` remains the source of
truth for the matching *algorithm* (chunking, distance reduction, action
priority); this document is the source of truth for *structure and API* —
how the code is organized, what the public API looks like, and the
RedisVL-specific operational constraints the implementation must respect.

## Goal

A Python-first guardrails service, built on RedisVL `SemanticRouter`, that is
easy to read, easy to test, and easy to extend — without sacrificing any of
the functional behavior described below (dual-stage routing, chunking for
long text, `INDETERMINATE` as a first-class outcome, and separately-measured
embedding/search performance).

No REST or GUI yet. REST should later be a thin wrapper around this API.

## Architecture

Three plain layers, no interfaces/protocols, no dependency-injection
wiring. Each file answers exactly one question:

```text
src/redis_guardrails/
├── models.py       # shared vocabulary: Guardrail, Chunk, Match, EvaluationResult, PerformanceInfo
├── errors.py       # EmbeddingError, SearchError, IncompleteCoverageError,
│                    # GuardrailNotFoundError, DuplicateGuardrailError, InvalidGuardrailError
├── chunking.py      # pure functions: split long text into overlapping chunks
├── store.py        # "how do we store/search" — owns both SemanticRouters; chunks + embeds + searches; raises on failure
├── evaluator.py     # "how do we decide" — pure, no I/O; reduces matches into one Decision
└── service.py       # "what does the caller see" — the public API; assembles search text; translates known errors into INDETERMINATE

tests/
├── fakes.py
├── test_chunking.py
├── test_evaluator.py
├── test_service.py
└── test_integration.py
```

Rejected alternative: a fully pluggable design (`VectorBackend`,
`Chunker`, `Embedder`, `DecisionPolicy` protocols, each with one real
implementation) was considered and rejected — it pays a real abstraction
cost (an interface file and a naming decision per concern) for
swappability nothing in this project currently needs. `store.py` already
isolates all RedisVL-specific code to one file; if a second backend is
ever needed, that's the only file that changes.

Why this split holds up as stages are added later: `store.py` and
`evaluator.py` are stage-agnostic today (store keys its router dict by
stage name; evaluator only sees `Match` lists). Adding a new stage costs
exactly one new method on `service.py` — never a change to `store.py` or
`evaluator.py` — because each stage is expected to need its own input
shape (e.g. a future `tool_call` stage would need `tool_name` +
`arguments`, not `text`), so a single generic `evaluate(stage, ...)`
entry point wouldn't actually avoid writing stage-specific code; it would
just move that special-casing into a loosely-typed `context` dict and
lose the type-signature guarantees below.

## Data Model (`models.py`)

```python
@dataclass
class Guardrail:
    id: str
    stage: Literal["input", "output"]
    category: str
    description: str
    examples: list[str]
    action: Literal["ALLOW", "FLAG", "BLOCK"]
    match_threshold: float

@dataclass
class Chunk:
    id: str
    source: Literal["input", "output"]
    start_character: int
    end_character: int
    text: str
    evaluated_text: str   # text as actually embedded (may include "User: ...\nAssistant: " prefix)

@dataclass
class Match:
    rule_id: str
    category: str
    action: Literal["ALLOW", "FLAG", "BLOCK"]
    distance: float
    threshold: float
    chunk_id: str
    evaluated_text: str

@dataclass
class PerformanceInfo:
    embedding_ms: float | None
    search_ms: float | None
    total_ms: float

@dataclass
class EvaluationResult:
    evaluation_id: str
    stage: Literal["input", "output"]
    status: Literal["COMPLETED", "INDETERMINATE"]
    action: Literal["ALLOW", "FLAG", "BLOCK"] | None
    primary_match: Match | None
    matches: list[Match] | None    # populated only when include_trace=True
    chunks: list[Chunk] | None      # populated only when include_trace=True
    performance: PerformanceInfo
```

## Public API (`service.py`)

```python
class GuardrailService:
    def add_guardrail(self, guardrail: Guardrail) -> None: ...
    def update_guardrail(self, guardrail: Guardrail) -> None: ...
    def delete_guardrail(self, guardrail_id: str) -> None: ...
    def get_guardrail(self, guardrail_id: str) -> Guardrail: ...
    def list_guardrails(self, stage: str | None = None) -> list[Guardrail]: ...

    def evaluate_input(
        self,
        text: str,
        include_trace: bool = False,
    ) -> EvaluationResult: ...

    def evaluate_output(
        self,
        response_text: str,
        request_text: str | None = None,
        include_trace: bool = False,
    ) -> EvaluationResult: ...
```

Design decisions on this surface:

- **Two named evaluation methods, not one generic `evaluate()`.** `evaluate_input` needs only `text`; `evaluate_output` needs `response_text` and optionally `request_text`. Separate methods let the signature enforce what each stage actually requires, instead of pushing a conditional requirement into a runtime check.
- **`evaluate_output`'s `request_text` is optional.** Accepted trade-off: some output categories are context-independent and detectable from the response alone (e.g. `harmful_content`, `bias`, overt tone violations like insults). Others — `goal_misalignment` above all — are relational by definition and cannot match without the request as context. Omitting `request_text` does not silently degrade to `ALLOW`; it simply means context-dependent categories will not be able to match, because nothing was provided for them to compare against. This is a documented limitation of the input, not an unsafe default.
- **CRUD methods raise, they don't return a status.** `add_guardrail`/`update_guardrail`/`delete_guardrail`/`get_guardrail` raise (`GuardrailNotFoundError`, `DuplicateGuardrailError`, `InvalidGuardrailError`) rather than returning an object the caller must branch on — CRUD failure is exceptional, not a normal outcome.
- **Evaluation methods never raise for expected failure modes.** Embedding/search failures and other conditions that prevent a safe decision surface as `status: INDETERMINATE`, not exceptions — this is the normal, expected shape of "we couldn't tell," not an error state for the caller to catch.

## Data Flow

### `evaluate_input(text, include_trace=False)`

1. `chunking.chunk_text(text)` → list of `Chunk` (whole text if it fits the embedding model's max input; overlapping chunks on paragraph/sentence/word boundaries otherwise, guaranteeing every character appears in at least one chunk).
2. `store.embed([c.evaluated_text for c in chunks])` — timed separately as `embedding_ms`. `evaluated_text` (not `text`) is what's embedded, since for `evaluate_output` it carries the `"User: ...\nAssistant: "` prefix; embedding `text` instead would make the `request_text` context a display-only no-op with zero effect on actual matching. Embedding-cache reads are bypassed here so this always reflects real embedding creation.
3. For each chunk vector, `store.search(vector, stage="input")` — queries the `input` `SemanticRouter` with `aggregation_method=DistanceAggregationMethod.min` explicitly set (RedisVL's default is `avg`, which would violate the "never average" rule), returning one `Match` per guardrail whose distance for this chunk is within its threshold. Timed separately as `search_ms`.
4. All per-chunk matches across all chunks are flattened and passed to `evaluator.decide(matches)`:
   - Reduce to the minimum distance per `rule_id` across all chunks (never averaged) — this is a second reduction beyond RedisVL's own per-chunk `min` aggregation across a route's examples, because a rule can also match in more than one chunk.
   - Keep the strongest (lowest-distance) match per `category`.
   - Choose the highest-priority action across matched categories (`BLOCK > FLAG > ALLOW`).
   - Choose the primary match from the guardrails carrying that action, breaking ties by greatest threshold margin, then stable guardrail ID.
   - No matches at all → `ALLOW`, "no detected semantic match".
5. `service.py` assembles the `EvaluationResult`: `status="COMPLETED"`, the decided `action`, `primary_match`, and (if `include_trace`) full `matches` and `chunks` with original text preserved.

### `evaluate_output(response_text, request_text=None, include_trace=False)`

Same as above, with one difference in step 1: only `response_text` is chunked. If `request_text` is provided, `"User: {request_text}\nAssistant: "` is prepended to every response chunk before embedding (per the Long Text rule: "split only the response and prepend the request to every chunk"). If `request_text` is omitted, chunks are embedded as-is with no prefix. Everything from step 2 onward is identical, searched against the `output` router.

### Note on reference-level matching (out of scope for now)

RedisVL's `RouteMatch` exposes only `name` (route/rule id) and `distance` —
never which specific example within that route produced the winning
distance. Recovering that would require a separate lower-level vector
query against the router's underlying index (filtered by the matched
`route_name`, sorted by distance, returning `reference_id`/`reference`)
rather than the router's own aggregated `route()`/`route_many()` call.
For now, the design assumes each chunk arrives with its per-guardrail
classification (rule/category/action/distance) already attached, and this
reference-level lookup is deferred rather than built.

## RedisVL Operational Constraints

- Two `SemanticRouter` instances, named `guardrails-input` and `guardrails-output`, give explicit stage isolation while `service.py` still exposes one unified API.
- `store.py` owns the Redis connections and both routers; it is the only file that constructs or reconstructs them.
- If a router is reconstructed via `from_existing()`, it must be reconstructed with its complete route configuration — never attach using an empty or partial local route list, and take extra care with custom vectorizers, since vectorizer configuration may not reconstruct correctly.
- RedisVL route mutations (`add`/`update`/`delete`) are not transactional. `store.py` must use best-effort cleanup/rollback on partial failure.
- Assume a single writer process for now — router route lists are cached locally, so concurrent writers from separate processes can drift out of sync.
- If an unrecoverable consistency error occurs, the recovery path is to reconstruct both routers from Redis, not to patch state in place.
- Verify the current appropriate RedisVL version before implementation (an earlier prototype needed `>=0.27.1`, since `0.20.0` lacked required dynamic route operations).

## Error Handling

| Condition | Surfaced as |
|---|---|
| Embedding provider failure | `store` raises `EmbeddingError` → `service` catches → `status="INDETERMINATE"`, `action=None` |
| Redis search/connection failure | `store` raises `SearchError` → `service` catches → `INDETERMINATE` |
| Text cannot be fully covered by chunks (e.g. a single token exceeds model max input) | `chunking`/`store` raises `IncompleteCoverageError` → `service` catches → `INDETERMINATE` |
| No guardrails configured for the requested stage | `store` raises (empty-router condition) → `service` catches → `INDETERMINATE` (never `ALLOW` — an unconfigured stage is not evidence of safety) |
| `get_guardrail`/`update_guardrail`/`delete_guardrail` on unknown id | `GuardrailNotFoundError` raised to caller |
| `add_guardrail` with a duplicate id | `DuplicateGuardrailError` raised to caller |
| `add_guardrail`/`update_guardrail` with invalid stage/action/threshold, or zero examples | `InvalidGuardrailError` raised to caller |

`evaluator.py` never raises for these conditions — it only ever runs on
already-valid `Match` lists. All failure translation happens in
`service.py`, which is the only layer that knows "a known error means
INDETERMINATE."

## Testing Approach

- **`test_evaluator.py`** — the highest-value tests. Pure Python `Match` lists in, `Decision` out. No Redis. Covers: category collapsing, action priority ordering, primary-match tie-breaking (margin then ID), empty-matches → ALLOW, multi-chunk min-reduction for the same rule.
- **`test_chunking.py`** — plain text fixtures. Covers: whole-text-fits case, overlap, boundary preference (paragraph > sentence > word), full-coverage guarantee, and the max-chunk-count limit triggering `IncompleteCoverageError`.
- **`test_service.py`** — `service.py` wired against a fake `store` (`tests/fakes.py`) to test the façade logic itself (search-text assembly for `evaluate_output` with/without `request_text`, error → `INDETERMINATE` translation, `include_trace` behavior) without needing real Redis.
- **`test_integration.py`** — runs against real Redis/RedisVL. There is intentionally no bulk loader in the public API — a loader is test/ops code that reads a file and calls `add_guardrail()` once per item, which permits individual additions, updates, deletions, and stage changes. This test loads `data/guardrails.json` that way and runs `data/testdata.json` (a separate set, never indexed) through the real service to measure pass/fail, category/action accuracy, false positives/negatives, and embedding/search performance measured separately.

## Explicitly Deferred

- Embedding provider and model.
- Redis deployment and authentication settings.
- Production threshold values (seed data uses `0.5` everywhere as a placeholder).
- Whether multi-process writers are eventually required (design assumes one writer process — see RedisVL Operational Constraints above).
- Reference/example-level match tracing (see "Note on reference-level matching" above).
- REST wrapper and GUI — future phases once the core API is implemented.
