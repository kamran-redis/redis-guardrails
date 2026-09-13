from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from redis_guardrails import GuardrailService
from redis_guardrails.chunking import DEFAULT_MAX_CHARS, DEFAULT_MAX_CHUNKS, DEFAULT_OVERLAP_CHARS
from redis_guardrails.cli.core import classify, run_benchmark, summarize_outcomes, summarize_performance
from redis_guardrails.web.deps import get_service
from redis_guardrails.web.templating import templates

router = APIRouter()

DATA_DIR = Path("data")


def _parse_optional_int(raw: str) -> int | None:
    raw = raw.strip()
    return int(raw) if raw else None


def _chunk_setting(value: int | None, default: int) -> dict:
    return {"value": value if value is not None else default, "is_default": value is None}


def _preset_files() -> list[str]:
    if not DATA_DIR.is_dir():
        return []
    return sorted(p.name for p in DATA_DIR.glob("*.json"))


@router.get("")
def benchmarks_form(request: Request):
    return templates.TemplateResponse(
        request,
        "benchmarks/index.html",
        {"cases": None, "performance": None, "outcomes": None, "classify": classify, "error": None},
    )


@router.post("/run")
async def run_benchmarks(
    request: Request,
    source: str = Form(...),
    preset_path: str = Form(default=""),
    upload_file: UploadFile | None = File(default=None),
    max_chars: str = Form(default=""),
    overlap_chars: str = Form(default=""),
    max_chunks: str = Form(default=""),
    service: GuardrailService = Depends(get_service),
):
    presets = _preset_files()
    tmp_path: Path | None = None

    try:
        if source == "preset":
            if preset_path not in presets:
                raise ValueError(f"unknown preset file: {preset_path!r}")
            path = DATA_DIR / preset_path
        else:
            if upload_file is None:
                raise ValueError("no file was uploaded")
            contents = await upload_file.read()
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="wb") as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(contents)
            path = tmp_path

        overrides = dict(
            max_chars=_parse_optional_int(max_chars),
            overlap_chars=_parse_optional_int(overlap_chars),
            max_chunks=_parse_optional_int(max_chunks),
        )
        cases = run_benchmark(service, path, **overrides)
    except KeyError as exc:
        return templates.TemplateResponse(
            request,
            "benchmarks/index.html",
            {"cases": None, "performance": None, "outcomes": None,
             "classify": classify,
             "error": f"case is missing required key {exc} (old input/output-key "
                      "benchmark files need migrating to the scope/text schema)"},
            status_code=400,
        )
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        return templates.TemplateResponse(
            request,
            "benchmarks/index.html",
            {"cases": None, "performance": None, "outcomes": None, "classify": classify, "error": str(exc)},
            status_code=400,
        )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    performance = summarize_performance(cases)
    outcomes = summarize_outcomes(cases)
    case_filters = {
        "scopes": sorted({c.scope for c in cases}),
        "categories": sorted({c.category or "n/a" for c in cases}),
        "actuals": sorted({c.result.action or "n/a" for c in cases}),
        "results": sorted({classify(c) for c in cases}),
    }
    chunk_settings = {
        "max_chars": _chunk_setting(overrides["max_chars"], DEFAULT_MAX_CHARS),
        "overlap_chars": _chunk_setting(overrides["overlap_chars"], DEFAULT_OVERLAP_CHARS),
        "max_chunks": _chunk_setting(overrides["max_chunks"], DEFAULT_MAX_CHUNKS),
    }
    return templates.TemplateResponse(
        request,
        "benchmarks/index.html",
        {"cases": cases, "performance": performance, "outcomes": outcomes,
         "case_filters": case_filters, "chunk_settings": chunk_settings,
         "classify": classify, "error": None},
    )
