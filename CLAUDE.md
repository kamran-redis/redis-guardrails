# redis_guardrails

A semantic guardrails service for LLM chatbots (built for a banking chatbot use case), built on [RedisVL](https://github.com/redis/redis-vl-python)'s `SemanticRouter`. Checks user requests (input) and model responses (output) against configurable guardrail rules using vector similarity, and decides `ALLOW`/`FLAG`/`BLOCK`.

See `README.md` for user-facing usage, `docs/vector-guardrail-algorithm.md` for the matching algorithm, and `docs/superpowers/specs/` + `docs/superpowers/plans/` for the design history of each subsystem (core API, CLI, web GUI) — each was built spec-first, then implemented from a task-by-task plan.

## Architecture (three layers, each reusing the one below unmodified)

1. **Core library** (`src/redis_guardrails/`) — `GuardrailService` (CRUD + `evaluate`), `GuardrailStore` (Redis/RedisVL persistence), `models.py` (`Guardrail`, `Match`, `Chunk`, `EvaluationResult`, `Scope`, `Action`), `errors.py`, `chunking.py`, `evaluator.py`. This is the only place business logic lives.
2. **CLI** (`src/redis_guardrails/cli/`) — `core.py` is a framework-agnostic layer (zero click, zero print, returns plain dataclasses) wrapping the core library; `formatting.py` is pure string presenters (zero click); `commands.py` is thin click glue. Entry point: `redis-guardrails`.
3. **Web GUI** (`src/redis_guardrails/web/`) — FastAPI + server-rendered Jinja2 templates, launched via `redis-guardrails serve`. Routes call `cli.core` functions and `GuardrailService` directly — **never reimplement evaluation/benchmark/CRUD logic in the web layer**. `_build_app()` (routers/static/exception-handlers, no service) is split from `create_app()` (adds the service via FastAPI lifespan) specifically so tests can exercise the whole app with a `FakeStore`, no real Redis or embedding model needed.

When extending any layer, put logic in the lowest layer that needs it and have the layers above just call it.

## Key invariants — don't casually change these

- `INDETERMINATE` beats a wrong guess: if evaluation can't complete safely (embedding failure, search failure, chunk-coverage failure), the result is `status="INDETERMINATE"`, never silently `ALLOW`.
- `EvaluationResult.matches`/`.chunks` are `None` unless the service was called with `include_trace=True`. Both `cli.core.evaluate_prompt` and `run_benchmark` always pass `include_trace=True` — display-layer trace gating (CLI's `--trace`, or "show chunks" in the GUI) is separate from whether the service *computed* the data.
- `Match.text` already exists directly on `Match` — never re-derive it by joining `Match.chunk_id` against a `Chunk`.
- Guardrail edit routes (web) take the guardrail id from the URL path, **never** from a form field — prevents a crafted POST from renaming a different guardrail via `{guardrail_id}/edit`.
- Benchmark preset file selection (web) validates the submitted filename against a **live glob** of `data/*.json` before ever calling `open()` — never trust a raw path string from a form (path-traversal guard).
- `redis_guardrails/cli/commands.py`: `import fastapi`/`import uvicorn`/`from redis_guardrails.web.app import create_app` must stay **inside** `serve_command`'s function body, never at module top-level — the module is imported unconditionally by the `redis-guardrails` console script, so a `.[cli]`-only install must not break `redis-guardrails --help`.
- Route registration order matters in FastAPI/Starlette: a static path (`/new`) must be registered before a dynamic path with the same prefix (`/{guardrail_id}`), or the static one gets swallowed.
- `Jinja2Templates.TemplateResponse` in the pinned dependency set requires `(request, name, context, ...)` with `request` as a required first positional argument — not the older `(name, {"request": request, ...})` form.
- Guardrail scopes (`Scope`, in `models.py`) are open strings, not a fixed `input`/`output` enum — a new scope's `SemanticRouter` is created lazily the first time a guardrail uses it (`GuardrailStore._ensure_router`). Known scopes are discovered live from Redis itself (`GuardrailStore._discover_scopes`, via `FT._LIST` filtered to the `guardrails-` index prefix) rather than a separate registry, so a second process — or a scope's index left over from before this process ever ran — is always found with no bookkeeping call needed on the write path. **A new scope name must not be a prefix of, or prefixed by, any existing scope name** — the Redis index key prefix is derived as `f"guardrails-{scope}"`, so e.g. `input` and `input2` would overlap and corrupt each other's data; `_ensure_router` rejects this. There is no default/fallback scope set — a fresh Redis with zero guardrails has zero known scopes.

## Environment

- Python 3.10+. Editable install: `.venv/bin/pip install -e ".[cli,web,test]"` (or a subset of extras as needed).
- `cli`/`embeddings` extras pull in `sentence-transformers` (real embedding model, ~90MB download on first use). `web` extra pulls in FastAPI/uvicorn/Jinja2/python-multipart — does **not** include `sentence-transformers`, so `redis-guardrails serve` needs `.[web,cli]` together, not `.[web]` alone.
- Requires **Redis Stack** (RediSearch module, not plain Redis) for anything that actually talks to Redis: `docker run -d -p 6379:6379 --name redis-stack-guardrails redis/redis-stack-server:latest`.
- Sandboxed environments here can raise permission errors on `pip install` (SSL cert access), on `git worktree remove` for uncommitted files, and on anything binding/connecting to a network port (`uvicorn`, `redis-cli`, integration tests hitting Redis) — these are sandbox restrictions, not real errors; retry with the sandbox disabled rather than treating them as project bugs.

## Testing

- `.venv/bin/pytest -m "not integration"` — fast suite, no Redis needed. Uses `tests/fakes.py::FakeStore` as the `GuardrailStore` substitute everywhere (CLI tests monkeypatch `build_service`; web tests use FastAPI's `app.dependency_overrides[get_service]`).
- `tests/test_web/` skips cleanly via `pytest.importorskip("fastapi")` when the `web` extra isn't installed — don't remove that guard.
- Full suite (real Redis + real embedding model): `REDIS_GUARDRAILS_ALLOW_TEST_OVERWRITE=1 .venv/bin/pytest -v` — the env var is a safety gate since some tests wipe the Redis index.
- When changing `cli/core.py` or `web/routes/*.py`, run the specific test file first, then the full non-integration suite once before committing — this project's history has caught real bugs (e.g. a Starlette signature mismatch, an unhandled `ValueError` on bad form input) exactly at that "run the whole suite" step.

## Conventions

- TDD: write the failing test, confirm it fails for the right reason, implement, confirm it passes.
- New guardrail/CRUD or evaluation features go in the core library first, with CLI and web as thin consumers — never add a route handler that duplicates logic already in `GuardrailService`/`cli.core`.
- Sample/test data lives in `data/` (`guardrails.json`, `testdata.json`, `testdata_extra.json`) — both the CLI's `benchmark` command and the web GUI's benchmark/prompt-picker features read from this directory via a live glob, so new files dropped in `data/*.json` become available automatically. Test-data cases use `text` (what's being checked) — not scope-specific `input`/`output` keys. There is no separate context/prior-turn field; a case that needs prior-turn context inlines it directly in `text`.
- No auth, no JS framework, no parallel REST/JSON API on the web GUI — it's an internal tool matching the CLI's own no-auth, local-Redis posture. Keep additions consistent with that unless explicitly asked to change it.
- Known bugs are tracked in `BUGS.md`, not-yet-scheduled feature/content ideas in `BACKLOG.md` — add to these when you find or propose one, rather than only raising it in conversation.
