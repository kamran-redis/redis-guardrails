# Open-Ended Stages — Design

## Status

Proposed design. Supersedes one decision in
`2026-09-10-redis-guardrails-core-api-design.md` (see "Relationship to the
original core API design" below) — everything else in that document still
holds.

## Vocabulary check (from conversation)

"Guardrail," in the product sense the user means it, is **one semantic
router** — today there are exactly two: the input router and the output
router. Each holds many individual rules (what the code calls `Guardrail`
objects / RedisVL `Route`s) inside it. This spec is about that router-level
concept: making the set of routers open-ended instead of hardcoded to
exactly `{"input", "output"}`.

## Goal

Today, `Stage = Literal["input", "output"]` (`models.py`). Adding a third
stage means changing code in five places (below). The goal: adding a new
stage should cost **zero code changes** — write a guardrail with
`"stage": "input2"` into the JSON data (or via the web form), and a new
router is created for it automatically, immediately usable.

## Non-goal / scope constraint (important)

The original core-API design deliberately rejected a single generic
`evaluate(stage, ...)` method, reasoning that a hypothetical future stage
(their example: `tool_call`, needing `tool_name` + `arguments`) would need
its own bespoke shape, and a generic method would just push that into a
loosely-typed dict.

That reasoning is still correct **in general** — but it's in tension with
"add a stage with zero code changes." You can't have both "any new stage
works with no code" and "a new stage might need an arbitrary new input
shape." This spec resolves the tension by narrowing scope deliberately:

> Every stage — present and future — is constrained to the same shape:
> **one piece of text to check, plus an optional piece of prior-turn
> context.** A stage needing a genuinely different shape (structured
> tool-call arguments, multiple attachments, etc.) is explicitly out of
> scope for "add it via JSON alone" — it would need its own method, exactly
> as the original design anticipated. This covers every stage in the
> project today (`input`, `output`) and any realistic near-term one
> (`input2`, a second bot's `output`, a `tool_response` stage that's still
> fundamentally "check this text against these rules").

## Relationship to the original core API design

`2026-09-10-redis-guardrails-core-api-design.md`'s "Public API" section
specified two named methods (`evaluate_input`, `evaluate_output`) instead of
one generic `evaluate()`. This spec replaces that with one generic method
(below), justified by the scope constraint above. Everything else in that
document — the four-layer split, `INDETERMINATE` semantics, chunking
behavior — is unchanged.

## Data model changes (`models.py`)

```python
Stage = str  # was: Literal["input", "output"]
```

`stage` values still get validated — not against a fixed set anymore, but
against the same kind of character-class pattern `Guardrail.id` already
uses (documented in `models.py`: unsafe characters break the RediSearch
filter Redis builds from route names). The reasoning doubles now: a stage
name also becomes part of a **Redis index name** (see below), so the same
validation protects two things at once.

```python
_VALID_STAGE_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
```

`validate_guardrail()`'s current `if guardrail.stage not in _VALID_STAGES`
check becomes a pattern match instead of a set-membership check.

## Store changes (`store.py`)

Today:
```python
_ROUTER_NAMES: dict[str, str] = {"input": "guardrails-input", "output": "guardrails-output"}
# __init__ eagerly builds exactly these two routers from this hardcoded dict
```

New:
```python
def _router_name(stage: Stage) -> str:
    return f"guardrails-{stage}"
```
Existing stages keep their exact same Redis index names (`guardrails-input`,
`guardrails-output`) — **no reindexing, no data migration for what's
already in Redis.**

**Discovering which routers to attach at startup.** There's no registry
today because there were only ever two, hardcoded. With an open set, the
store needs to know what already exists in Redis before it can attach to
it. Proposal: a small persisted Redis SET at a fixed key
(`guardrails:stages`) listing every stage name ever used. `__init__` reads
this set and attaches a router for each entry found (falling back to
`{"input", "output"}` if the set doesn't exist yet, so a fresh Redis
instance behaves exactly as it does today). `add()` adds the new stage name
to this set (`SADD`) the first time it sees one not already in
`self._routers`, alongside creating the router itself.

**Lazy router creation.** `add()`, `get()`, `list()`, `search()`, `delete()`,
`update()` currently do `self._routers[stage]` / `self._routers[guardrail.stage]`
assuming pre-population — this raises `KeyError` for an unseen stage today.
They change to a shared `_get_or_create_router(stage)` helper that creates
and caches (via `_attach_or_create`, unchanged) a router on first use.

**Interaction with the already-open staleness bug** (`BUGS.md`): a
long-running web process only knows about stages it discovered at its own
startup. If a brand-new stage is added via the CLI while the web app is
running, that process won't have it in `self._routers` *or* know to look —
today's bug ("new guardrails invisible until restart") generalizes to
"a whole new stage can be invisible until restart," which is a bigger miss
than before. **Recommend fixing the registry-refresh gap in `BUGS.md`
together with this change**, not after — otherwise this change ships a
sharper version of a bug already on record.

## Service changes (`service.py`)

```python
def evaluate(
    self,
    stage: str,
    text: str,
    context: str | None = None,
    include_trace: bool = False,
    max_chars: int | None = None,
    overlap_chars: int | None = None,
    max_chunks: int | None = None,
) -> EvaluationResult:
    prefix = f"User: {context}\nAssistant: " if context is not None else ""
    return self._evaluate(stage=stage, text=text, prefix=prefix, ...)
```

Replaces `evaluate_input(text)` and `evaluate_output(response_text, request_text)`.
`context` (was `request_text`, output-only) is now available for **any**
stage — this is the generalization of the "User: X / Assistant: Y" shape
that's already how `evaluate_output` works internally today, just not
exposed as a first-class idea.

**Breaking change, both call sites in this repo:**
- `evaluate_input(text)` → `evaluate("input", text)`
- `evaluate_output(response_text, request_text=r)` → `evaluate("output", response_text, context=r)`

The two named methods are removed outright, no deprecated wrappers — this
is a pre-1.0 internal tool, every caller lives in this same repo (CLI, web,
tests), and there's no backward compatibility to preserve.

## CLI changes (`cli/`)

- `evaluate_group`'s two subcommands (`evaluate input`, `evaluate output`)
  collapse into one `evaluate` command: `--stage TEXT` (free text, required),
  `--text`, `--context` (optional). This is a **breaking CLI surface
  change** — worth calling out to anyone with scripts using the old
  subcommands.
- `cli/core.py`'s `evaluate_prompt_input`/`evaluate_prompt_output` collapse
  into one `evaluate_prompt(service, stage, text, context=None, **overrides)`.
- `run_benchmark`'s branch (`cli/core.py` ~line 127) currently does:
  ```python
  if case["stage"] == "input":
      result = service.evaluate_input(case["input"], ...)
  else:
      result = service.evaluate_output(case["output"], request_text=case.get("input"), ...)
  ```
  This hardcodes which JSON keys hold the text based on the literal string
  `"input"` — it has no principled way to handle a novel stage name. It
  needs the test-data JSON shape to generalize too (next section), then
  becomes simply:
  ```python
  result = service.evaluate(case["stage"], case["text"], context=case.get("context"), ...)
  ```

## Test-data format changes (`data/testdata*.json`)

Current shape (stage-specific keys):
```json
{"id": "x", "stage": "output", "input": "customer question", "output": "bot reply", "category": "...", "action": "..."}
```
New shape (stage-agnostic keys, matches the new `evaluate()` signature
directly):
```json
{"id": "x", "stage": "output", "text": "bot reply", "context": "customer question", "category": "...", "action": "..."}
```
For `input`-stage cases, `context` is simply omitted. This is a real
migration of `data/testdata.json` and `data/testdata_extra.json` (76 cases
total) — a one-time, scriptable rename, no compatibility shim for the old
key names.

`web/routes/prompts.py`'s `_load_test_cases()` picks up the new keys
directly (`case["text"]`, `case.get("context")`) instead of its current
`if stage == "output": text, request_text = case["output"], case.get("input", "")`
branch.

## Web changes (`web/`)

- **Guardrail form** (`guardrails/form.html`): the hardcoded
  `<option value="input">` / `<option value="output">` `<select>` becomes a
  free-text input with a `<datalist>` populated from the distinct stage
  values `service.list_guardrails()` already returns — typeable (so a new
  stage works) but suggests existing ones (so you don't typo `imput`).
- **Guardrail list filter** (`guardrails/list.html`): the hardcoded
  All/Input/Output tabs become one tab per distinct stage actually present,
  computed the same way, plus "All".
- **Evaluate page**: same dynamic-list treatment for its stage picker
  (this also resolves the earlier "why two tabs" friction — the list simply
  reflects however many stages exist, not a hardcoded binary choice). The
  "request context" field stops being gated to one hardcoded stage name and
  becomes a plain optional field available for any stage, matching the
  generalized `context` parameter.

## Testing

- `validate_guardrail` rejects a stage name with unsafe characters, accepts
  one that just isn't `"input"`/`"output"`.
- `GuardrailStore`: adding a guardrail with a brand-new stage in one call
  creates a usable router immediately (same process) — searchable right
  after, no restart.
- `GuardrailStore.__init__` against a Redis instance that already has a
  third stage's index (simulating a second process's earlier write)
  attaches to it correctly via the stages registry.
- `GuardrailService.evaluate("input2", "some text")` round-trips end to end
  against a `FakeStore` configured for a non-`input`/`output` stage.
- CLI: `redis-guardrails evaluate --stage input2 --text "..."` works.
- Web: submitting the guardrail form with a typed-in new stage creates a
  guardrail on that stage; the evaluate page's stage list picks it up.
- `run_benchmark` against a test-data file using a novel stage name.

## Decided (no backward compatibility needed)

- `evaluate_input`/`evaluate_output` are **removed outright** — no
  deprecated wrappers, no transition window. Every in-repo caller (CLI,
  web, tests) is updated to call `evaluate(stage, text, context=...)`
  directly.
- `data/testdata.json` / `data/testdata_extra.json` get a **real migration**
  to the `text`/`context` key shape — no compatibility shim reading both
  old and new shapes. There is no external consumer of these files to stay
  compatible with.

## Open decisions still to confirm

1. Stages registry as an explicit Redis SET (recommended — version-stable,
   doesn't depend on RediSearch introspection commands) vs. discovering
   existing routers by scanning for `guardrails-*` indices directly.
2. Fix the `BUGS.md` router-staleness bug together with this change
   (recommended, since this change makes it worse) vs. ship this
   independently and fix that bug separately later.
