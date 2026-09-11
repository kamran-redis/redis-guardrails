# redis_guardrails CLI — Design

## Status

Approved design. This document is the source of truth for the CLI's structure and behavior. The core `redis_guardrails` API it wraps (`GuardrailService`, `GuardrailStore`, models, errors) is unchanged — see `docs/superpowers/specs/2026-09-10-redis-guardrails-core-api-design.md`.

## Context

The core API is done and merged to `main` (53 tests passing). There's no way to use it except by writing Python or running pytest. This adds a command-line interface to (1) load a guardrails JSON file into Redis, and (2) run test data through the service as a benchmark, reporting pass/fail and performance. A GUI is planned for later ("browse the data, run prompts, see results") — so the CLI's underlying logic must be reusable by that future GUI, not buried inside command-line-parsing callbacks.

Decisions already made:
- Framework: **click**
- Commands: **load**, **benchmark**, and **evaluate** (ad-hoc single-prompt eval)
- Benchmark output: **human-readable table only** (no JSON mode for now)

## Architecture

New sub-package, mirroring the core API's existing "plain layers" pattern (store/evaluator/service):

```
src/redis_guardrails/cli/
├── __init__.py     # exports `cli` (the click Group) for the entry point
├── core.py         # framework-agnostic logic. Zero `import click`, zero print(). Returns plain
│                    # dataclasses. This is what a future GUI imports directly.
├── formatting.py    # pure str-returning presenters. Zero `import click`.
└── commands.py      # click Group + the 3 commands — thin glue: parse args, call core.py,
                       # call formatting.py, click.echo() the result, set exit code.
```

`core.py` never catches "shouldn't happen" errors (bad Redis URL, missing model) — those propagate so every caller decides how to present them. It does catch expected per-item failures during `load`, because that's real business behavior a GUI needs too, not a CLI-only concern.

## Shared service construction (`core.py`)

```python
DEFAULT_REDIS_URL = "redis://localhost:6379"               # matches tests/conftest.py
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"     # matches every existing usage

def build_service(redis_url: str = DEFAULT_REDIS_URL, model: str = DEFAULT_MODEL, overwrite: bool = False) -> GuardrailService:
    vectorizer = HFTextVectorizer(model=model)
    store = GuardrailStore(redis_url=redis_url, vectorizer=vectorizer, overwrite=overwrite)
    return GuardrailService(store)
```

`commands.py` exposes this via `--redis-url` (envvar `REDIS_URL`) and `--model` (envvar `REDIS_GUARDRAILS_MODEL`) options with these same defaults.

## `load` command

Mirrors the existing test loader (`tests/test_integration.py::_load_guardrails`) but **skips and continues** on a per-item error instead of aborting — one bad record (duplicate id, out-of-range threshold) shouldn't block the rest. Exit code is non-zero if any errors occurred.

```python
@dataclass
class LoadItemError:
    index: int
    guardrail_id: str | None
    error: Exception

@dataclass
class LoadReport:
    total: int
    added: list[str]
    errors: list[LoadItemError]

def load_guardrails_from_file(service: GuardrailService, path: Path) -> LoadReport: ...
```

CLI: `redis-guardrails load PATH [--redis-url URL] [--model MODEL] [--overwrite]`. `--overwrite` is an explicit flag a human types deliberately — sufficient safeguard for a CLI (the test suite's `REDIS_GUARDRAILS_ALLOW_TEST_OVERWRITE` env-var gate exists specifically to stop the *automated test suite* from silently wiping a real Redis; a deliberate human CLI invocation doesn't need that extra layer).

## `benchmark` command

Runs every case in a testdata file through `evaluate_input`/`evaluate_output` with `include_trace=True` (free — the service always computes matches/chunks; trace only gates whether they're attached), and reports pass/fail, a category breakdown, false-positive/negative/wrong-severity counts, and performance.

```python
@dataclass
class CaseResult:
    case_id: str
    stage: Stage
    category: str
    expected_action: Action
    result: EvaluationResult

def run_benchmark(service: GuardrailService, path: Path) -> list[CaseResult]: ...

def classify(case: CaseResult) -> str:
    """Returns one of: PASS, FALSE_POSITIVE, FALSE_NEGATIVE, WRONG_SEVERITY, INDETERMINATE."""

@dataclass
class PerformanceSummary:
    count: int
    avg_embedding_ms: float | None   # None only if every case was INDETERMINATE
    avg_search_ms: float | None
    avg_total_ms: float               # always populated
    p95_total_ms: float
    indeterminate_count: int

def summarize_performance(cases: list[CaseResult]) -> PerformanceSummary: ...
```

CLI: `redis-guardrails benchmark PATH [--redis-url URL] [--model MODEL] [--min-accuracy FLOAT]`. `--min-accuracy` is optional (default `None` = report only); if set, exit code is `1` when accuracy falls below it.

## `evaluate` command

A click **group** with two subcommands, matching `evaluate_input`/`evaluate_output` on the service:

```
redis-guardrails evaluate input TEXT [--redis-url URL] [--model MODEL] [--trace]
redis-guardrails evaluate output --response TEXT [--request TEXT] [--redis-url URL] [--model MODEL] [--trace]
```

`--trace` adds the full match list and chunk detail (showing `evaluated_text` alongside `text` — the only way to see the `"User: ...\nAssistant: ..."` prefix). `INDETERMINATE` is a normal result for this command, not an error — exits `0`.

## pyproject.toml changes

- `click>=8.1` added to core `dependencies` (tiny, zero transitive deps, needed for `--help` to work at all).
- New `cli = ["sentence-transformers"]` extra (alias of the existing `embeddings` extra) — `pip install -e ".[cli]"` is the documented "everything needed to run the CLI" command.
- `[project.scripts]` `redis-guardrails = "redis_guardrails.cli:cli"`.
- `build_service()` raises a clear `RuntimeError` pointing at `[cli]` if `sentence-transformers` isn't installed.

## Testing

- `tests/test_cli/test_core.py` — pure unit tests against `cli.core` using `GuardrailService(FakeStore())` — no Redis, no click.
- `tests/test_cli/test_formatting.py` — string-assertion tests against the formatters, hand-built dataclasses.
- `tests/test_cli/test_commands.py` — `click.testing.CliRunner`, monkeypatching `redis_guardrails.cli.commands.build_service` to return a `FakeStore`-backed service.
- `tests/test_cli/test_cli_integration.py`, `@pytest.mark.integration`, reusing `tests/conftest.py`'s `redis_url`/`allow_test_overwrite` fixtures — one real end-to-end smoke test.
