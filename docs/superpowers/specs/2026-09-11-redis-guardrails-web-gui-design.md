# Web GUI for redis_guardrails — Design

## Context

The `redis_guardrails` project has a working core library (`GuardrailService`/`GuardrailStore`, `docs/superpowers/specs/2026-09-10-redis-guardrails-core-api-design.md`) and CLI (`load`/`benchmark`/`evaluate`, `docs/superpowers/specs/2026-09-11-redis-guardrails-cli-design.md`), both already merged. Both prior specs state a GUI is planned but not built, and that `redis_guardrails.cli.core` was deliberately designed as a framework-agnostic layer specifically so a future GUI could reuse it directly rather than reimplementing evaluation/benchmark logic. This spec is that GUI.

**Scope, as requested by the user:** all three CLI capabilities available in a browser — running prompts (`evaluate input`/`evaluate output`), running benchmarks, and full guardrail CRUD (add/update/delete/get/list — `GuardrailService` already supports all of this, it's just never had a UI).

**Decided:** FastAPI + server-rendered Jinja2 HTML. No auth (matches the CLI's current local/internal posture — Redis Stack with no auth, no TLS). No web framework or GUI code exists anywhere in the repo yet; this is a net-new subsystem.

**Deviation from the earlier architecture sketch:** that sketch mentioned vendoring HTMX for delete-confirmation. Vendoring a real third-party minified JS file isn't something an implementation plan can embed inline, and it's overkill for the one thing it was for. This spec instead uses a plain `onsubmit="return confirm(...)"` attribute on the delete `<form>` — same UX (a confirm dialog before delete), zero new dependency, no vendored file.

## Package layout

```
src/redis_guardrails/web/
├── __init__.py
├── app.py                      # _build_app() + create_app()
├── deps.py                     # get_service(request) -> GuardrailService
├── templating.py               # shared Jinja2Templates instance
├── errors.py                   # register_exception_handlers(app)
├── routes/
│   ├── __init__.py
│   ├── home.py                 # GET /
│   ├── prompts.py               # GET/POST /prompts/evaluate
│   ├── benchmarks.py            # GET /benchmarks, POST /benchmarks/run
│   └── guardrails.py            # full CRUD, prefix /guardrails
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── error.html
│   ├── prompts/run.html
│   ├── benchmarks/index.html
│   └── guardrails/{list,detail,form}.html
└── static/
    └── css/style.css
```

## Service wiring

```python
# app.py
def _build_app() -> FastAPI:
    """Wires routers/static/exception handlers only — no service, no lifespan.
    Shared by create_app() and tests, so route registration never duplicates."""

def create_app(redis_url: str | None = None, model: str | None = None, overwrite: bool = False) -> FastAPI:
    """Wraps _build_app() with a lifespan that calls cli.core.build_service(...)
    once at startup and stores it on app.state.service."""

# deps.py
def get_service(request: Request) -> GuardrailService:
    return request.app.state.service
```

`build_service` constructs an `HFTextVectorizer` (loads a real embedding model) and `GuardrailStore.__init__` opens Redis connections eagerly — both must happen once at server startup, never per-request. A bad `--redis-url` fails loudly at `serve` startup, never silently on first page view.

Splitting `_build_app()` from `create_app()` is what makes testing possible without real Redis/a real model: tests call `_build_app()` directly and use FastAPI's `app.dependency_overrides[get_service] = lambda: service` to inject a `GuardrailService(FakeStore())` (`tests/fakes.py`, already exists, used unmodified — same fake the CLI tests use, substituted via DI override instead of monkeypatching `build_service`).

## Launch mechanism

A new `serve` subcommand on the existing `redis-guardrails` CLI (`cli/commands.py`), not a second console script — reuses the existing `--redis-url`/`--model` options and `REDIS_URL`/`REDIS_GUARDRAILS_MODEL` envvars, adds `--host` (default `127.0.0.1`) and `--port` (default `8000`).

```python
@cli.command("serve")
@_handle_errors
@_redis_url_option
@_model_option
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8000, show_default=True)
def serve_command(redis_url: str, model: str, host: str, port: int):
    """Launch the web GUI."""
    try:
        import uvicorn
        from redis_guardrails.web.app import create_app
    except ImportError as exc:
        raise click.ClickException(
            "the web GUI requires extra dependencies. Install with `pip install -e '.[web]'`."
        ) from exc
    app = create_app(redis_url=redis_url, model=model)
    click.echo(f"Starting redis_guardrails web GUI at http://{host}:{port} (Redis: {redis_url}, model: {model})")
    uvicorn.run(app, host=host, port=port)
```

The `import fastapi`/`uvicorn` stay **inside the function body** — `cli/commands.py` is imported unconditionally by the `redis-guardrails` console script, so users who only installed `.[cli]` must not get an `ImportError` just from running `redis-guardrails --help`.

## Dependencies

`pyproject.toml` gets a new optional-dependency group:
```toml
web = ["fastapi>=0.115", "uvicorn[standard]>=0.30", "jinja2>=3.1", "python-multipart>=0.0.9"]
```
(`python-multipart` is required by Starlette for `Form(...)`/`UploadFile` parsing.) The existing `test` extra gets `httpx` added (needed by `fastapi.testclient.TestClient`). Add package-data so templates/static ship with the installed package:
```toml
[tool.setuptools]
include-package-data = true
[tool.setuptools.package-data]
redis_guardrails = ["web/templates/**/*.html", "web/static/**/*"]
```

## Routes

All routes render HTML (`Jinja2Templates.TemplateResponse`); none return JSON — no parallel REST API in this scope.

### Run Prompts
- `GET /prompts/evaluate` → `prompts/run.html`, empty form.
- `POST /prompts/evaluate` → same template with `result` in context. Form fields: `stage` (`input`|`output`), `text` (required), `request_text` (optional, output-only), `max_chars`/`overlap_chars`/`max_chunks` (optional ints). Calls `evaluate_prompt_input(service, text, ...)` or `evaluate_prompt_output(service, text, request_text=request_text, ...)` from `cli.core` — both already run with `include_trace=True` unconditionally. Renders: status, action, primary match, full matches table (`rule_id`/`category`/`action`/`distance`/`threshold`/`chunk_id`/`evaluated_text`), and the chunk breakdown (char ranges + evaluated text + that chunk's own matches — mirrors CLI `--trace`).

### Run Benchmarks
- `GET /benchmarks` → `benchmarks/index.html`, form only.
- `POST /benchmarks/run` → same template with `cases`, `performance` in context. Form fields: `source` (`preset`|`upload`), `preset_path` (validated against a **live** `Path("data").glob("*.json")` result — never trust a raw path string from the form, closing off path traversal), `upload_file` (optional `UploadFile`), chunking overrides. Uploaded bytes go to a `tempfile.NamedTemporaryFile(suffix=".json", delete=False)` cleaned up in a `finally`, so `run_benchmark(service, path, ...)` from `cli.core` is called completely unmodified. Calls `run_benchmark(...)` then `summarize_performance(cases)`. Malformed JSON/case shape (`json.JSONDecodeError`/`KeyError`/`TypeError`) → caught, re-render with `error=str(exc)`, HTTP 400, never a raw traceback.

### Guardrail CRUD (prefix `/guardrails`)
- `GET /guardrails?stage=input|output` → `service.list_guardrails(stage=stage)` → `guardrails/list.html`.
- `GET /guardrails/new` → `guardrails/form.html`, mode=`"create"`.
- `POST /guardrails/new` → build `Guardrail(...)`, call `service.add_guardrail(...)`. Success → 303 redirect to `GET /guardrails/{id}`. `InvalidGuardrailError`/`DuplicateGuardrailError` → re-render form, mode=`"create"`, HTTP 400, submitted values preserved, `error=str(exc)`.
- `GET /guardrails/{id}` → `service.get_guardrail(id)` → `guardrails/detail.html`; `GuardrailNotFoundError` → `error.html`, 404.
- `GET /guardrails/{id}/edit` → same lookup/404 → `guardrails/form.html`, mode=`"edit"`, id field read-only.
- `POST /guardrails/{id}/edit` → handler **forces `guardrail.id = id` from the URL path, never a form field** (prevents a crafted request from renaming a different guardrail). Calls `service.update_guardrail(...)`. Success → 303 to detail. `InvalidGuardrailError` → re-render form, 400. `GuardrailNotFoundError` (deleted concurrently) → `error.html`, 404.
- `POST /guardrails/{id}/delete` → `service.delete_guardrail(id)` → 303 to `GET /guardrails?flash=deleted:{id}`. `GuardrailNotFoundError` (double-click/stale tab) → redirect to `?flash=not_found:{id}` rather than erroring — a repeat delete click should feel like a no-op.

Flash messages are a plain querystring value (`?flash=deleted:g-1`) read once by `list.html` — no cookies, no server-side session state, consistent with no-auth scope.

## Error handling (`web/errors.py`)

`register_exception_handlers(app)` maps:
- `GuardrailNotFoundError` → `error.html`, 404.
- `RedisError` (redis-py) → `error.html`, 502, "the guardrails store is unreachable."
- `RequestValidationError` (Starlette, e.g. missing required form field) → `error.html`, 422.
- Anything else uncaught → FastAPI's default 500 behavior (don't mask real bugs).

Note: `IncompleteCoverageError`/`EmbeddingError`/`SearchError` are already caught inside `GuardrailService._evaluate` and turned into `status="INDETERMINATE"` results — they never propagate out of `evaluate_input`/`evaluate_output`, so no handler is needed for them.

## Templates — content sketch

- `base.html` — head with `static/css/style.css` + `static/js/htmx.min.js`; nav: Home | Run Prompt | Benchmarks | Guardrails; `{% block content %}`.
- `home.html` — three link cards to the three features.
- `error.html` — status heading, `{{ message }}`, link home.
- `prompts/run.html` — stage radio, `text` textarea, `request_text` textarea, collapsible chunking-override fields, Evaluate button; results section as described above.
- `benchmarks/index.html` — source radio (preset select / file upload), chunking overrides, Run button; results: summary counts, category-breakdown table, performance panel, per-case table.
- `guardrails/list.html` — stage filter links, "+ New guardrail", flash banner, table with View/Edit/Delete (delete is a small `<form>` with `onsubmit="return confirm('Delete guardrail ...?')"`).
- `guardrails/detail.html` — full field display, Edit/Delete(same confirm form)/Back.
- `guardrails/form.html` — shared create/edit; id editable only in create mode; stage select; category/description/examples (one per line) fields; action select; `match_threshold` number input (`min="0.01" max="2.0" step="0.01"` — the *exclusive* lower bound matches `validate_guardrail`); inline error banner.

All writes use plain POST+redirect (PRG pattern) — `add_guardrail`/`update_guardrail` are not idempotent, so avoiding accidental double-submit-on-refresh matters more than snappiness. Run Prompt/Run Benchmark are plain full-page POST re-renders — both are read-only/idempotent, no correctness risk in resubmission. No JS framework, no vendored library — the one bit of interactivity (delete confirmation) is a single inline `onsubmit` attribute.

## Explicitly out of scope (YAGNI)

Pagination (9 guardrails, ≤65 benchmark cases), auth/sessions, a parallel JSON/REST API, websockets/streaming for benchmark progress, multi-user edit-conflict handling, a JS bundler/framework, CSRF tokens (nothing to protect without auth), bulk-import via browser (CLI's `load` already covers it), dynamic JS example-list editor, dark mode/mobile layout.

## Testing

`tests/test_web/` — `conftest.py` starts with `pytest.importorskip("fastapi")` so the suite skips cleanly when `.[web]` isn't installed (same pattern as existing Redis-availability skip-gates), plus `service` (`GuardrailService(FakeStore())`) and `client` (`TestClient(_build_app())` with `get_service` overridden) fixtures. `test_prompts.py`, `test_benchmarks.py`, `test_guardrails.py` — each covers a success path and at least one error path (duplicate-id create, invalid threshold, malformed benchmark JSON, unknown preset path, missing-text prompt submission, delete-then-404).

## Verification

1. `pip install -e ".[web,cli,test]"` then `pytest tests/test_web/ -v` — new tests pass.
2. `pytest -m "not integration"` — full suite green, and passes even without `.[web]` installed.
3. With Redis Stack running: `redis-guardrails serve`, manually exercise all three features — prompts (input/output, with/without request context), benchmarks (preset and upload), full guardrail create/edit/delete cycle, confirming error cases show clean messages, not tracebacks.
4. Update `README.md`: move the GUI bullet from 🚧 to ✅, add a `serve` usage example.
