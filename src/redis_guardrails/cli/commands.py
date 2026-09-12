from __future__ import annotations

import functools
from pathlib import Path

import click
from redis.exceptions import RedisError

from redis_guardrails.cli.core import (
    DEFAULT_MODEL,
    DEFAULT_REDIS_URL,
    build_service,
    classify,
    evaluate_prompt,
    load_guardrails_from_file,
    run_benchmark,
    summarize_performance,
)
from redis_guardrails.cli.formatting import (
    format_benchmark_report,
    format_evaluation_result,
    format_load_report,
)
from redis_guardrails.errors import GuardrailError

_redis_url_option = click.option(
    "--redis-url",
    envvar="REDIS_URL",
    default=DEFAULT_REDIS_URL,
    show_default=True,
    show_envvar=True,
    help="Redis connection URL (requires Redis 8+ for search/vector support).",
)
_model_option = click.option(
    "--model",
    envvar="REDIS_GUARDRAILS_MODEL",
    default=DEFAULT_MODEL,
    show_default=True,
    show_envvar=True,
    help="Embedding model name (passed to HFTextVectorizer).",
)
_max_chars_option = click.option(
    "--max-chars",
    type=int,
    default=None,
    help="Max characters per chunk before text is split (default: chunking.DEFAULT_MAX_CHARS, 800).",
)
_overlap_chars_option = click.option(
    "--overlap-chars",
    type=int,
    default=None,
    help="Characters of overlap between consecutive chunks (default: chunking.DEFAULT_OVERLAP_CHARS, 100).",
)
_max_chunks_option = click.option(
    "--max-chunks",
    type=int,
    default=None,
    help="Max number of chunks before evaluation gives up as INDETERMINATE (default: chunking.DEFAULT_MAX_CHUNKS, 50).",
)


def _handle_errors(command):
    @functools.wraps(command)
    def wrapper(*args, **kwargs):
        try:
            return command(*args, **kwargs)
        except KeyError as exc:
            raise click.ClickException(
                f"case is missing required key {exc} (old input/output-key benchmark "
                "files need migrating to the scope/text schema)"
            ) from exc
        except (RedisError, RuntimeError, GuardrailError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

    return wrapper


@click.group()
def cli():
    """redis_guardrails command-line interface."""


@cli.command("load")
@_handle_errors
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_redis_url_option
@_model_option
@click.option("--overwrite", is_flag=True, help="Wipe the existing index before loading (destructive).")
def load_command(path: Path, redis_url: str, model: str, overwrite: bool):
    """Load guardrails from a JSON file into the store."""
    click.echo(f"Loading guardrails from {path} ...")
    service = build_service(redis_url=redis_url, model=model, overwrite=overwrite)
    click.echo(f"Using Redis at {redis_url} (model: {model})\n")

    report = load_guardrails_from_file(service, path)
    click.echo(format_load_report(report))

    if report.errors:
        raise SystemExit(1)


@cli.command("benchmark")
@_handle_errors
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_redis_url_option
@_model_option
@click.option(
    "--min-accuracy",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Exit non-zero if action accuracy falls below this fraction (0.0-1.0).",
)
@_max_chars_option
@_overlap_chars_option
@_max_chunks_option
def benchmark_command(
    path: Path,
    redis_url: str,
    model: str,
    min_accuracy: float | None,
    max_chars: int | None,
    overlap_chars: int | None,
    max_chunks: int | None,
):
    """Run a test-data file through the service and report pass/fail and performance."""
    click.echo(f"Benchmark: {path}")
    service = build_service(redis_url=redis_url, model=model, overwrite=False)
    click.echo(f"Redis: {redis_url}   Model: {model}\n")

    cases = run_benchmark(
        service, path, max_chars=max_chars, overlap_chars=overlap_chars, max_chunks=max_chunks
    )
    performance = summarize_performance(cases)
    click.echo(format_benchmark_report(cases, performance))

    if min_accuracy is not None:
        passed = sum(1 for c in cases if classify(c) == "PASS")
        accuracy = (passed / len(cases)) if cases else 0.0
        if accuracy < min_accuracy:
            click.echo(f"\naccuracy {accuracy:.1%} is below the required {min_accuracy:.1%}")
            raise SystemExit(1)


@cli.command("evaluate")
@_handle_errors
@click.option("--scope", required=True, help="Which guardrail scope to check against (e.g. input, output).")
@click.argument("text")
@_redis_url_option
@_model_option
@click.option("--trace", is_flag=True, help="Show full match and chunk detail.")
@_max_chars_option
@_overlap_chars_option
@_max_chunks_option
def evaluate_command(
    scope: str,
    text: str,
    redis_url: str,
    model: str,
    trace: bool,
    max_chars: int | None,
    overlap_chars: int | None,
    max_chunks: int | None,
):
    """Evaluate a single prompt against one guardrail scope."""
    service = build_service(redis_url=redis_url, model=model, overwrite=False)
    result = evaluate_prompt(
        service, scope, text,
        max_chars=max_chars, overlap_chars=overlap_chars, max_chunks=max_chunks,
    )
    click.echo(format_evaluation_result(result, trace=trace))


@cli.command("serve")
@_handle_errors
@_redis_url_option
@_model_option
@click.option("--host", default="127.0.0.1", show_default=True, help="Host interface to bind the web server to.")
@click.option("--port", type=int, default=8000, show_default=True, help="Port to bind the web server to.")
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
