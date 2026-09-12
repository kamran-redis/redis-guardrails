from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request

from redis_guardrails import GuardrailService
from redis_guardrails.cli.core import evaluate_prompt
from redis_guardrails.web.deps import get_service
from redis_guardrails.web.templating import templates

router = APIRouter()

DATA_DIR = Path("data")


def _parse_optional_int(raw: str) -> int | None:
    raw = raw.strip()
    return int(raw) if raw else None


def _load_test_cases() -> list[dict]:
    """Flatten every case from every data/*.json file for the "load from
    test data" picker. Malformed files/cases are skipped, not fatal —
    this is a convenience picker, not the benchmark run itself."""
    cases: list[dict] = []
    if not DATA_DIR.is_dir():
        return cases
    for path in sorted(DATA_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(raw, list):
            continue
        for case in raw:
            if not isinstance(case, dict):
                continue
            try:
                stage = case["stage"]
                text = case["text"]
                case_id = case["id"]
            except KeyError:
                continue
            cases.append({
                "file": path.name,
                "id": case_id,
                "stage": stage,
                "category": case.get("category") or "other",
                "text": text,
                "context": case.get("context") or "",
            })
    cases.sort(key=lambda c: (c["category"], c["file"], c["id"]))
    return cases


@router.get("/evaluate")
def evaluate_form(request: Request, service: GuardrailService = Depends(get_service)):
    return templates.TemplateResponse(
        request,
        "prompts/run.html",
        {
            "result": None, "stage": "", "text": "", "context": "", "error": None,
            "test_cases": _load_test_cases(), "stages": service.known_stages(),
        },
    )


@router.post("/evaluate")
def evaluate_submit(
    request: Request,
    stage: str = Form(...),
    text: str = Form(...),
    context: str = Form(default=""),
    max_chars: str = Form(default=""),
    overlap_chars: str = Form(default=""),
    max_chunks: str = Form(default=""),
    service: GuardrailService = Depends(get_service),
):
    try:
        overrides = dict(
            max_chars=_parse_optional_int(max_chars),
            overlap_chars=_parse_optional_int(overlap_chars),
            max_chunks=_parse_optional_int(max_chunks),
        )
        result = evaluate_prompt(service, stage, text, context=context or None, **overrides)
    except ValueError as exc:
        return templates.TemplateResponse(
            request,
            "prompts/run.html",
            {
                "error": str(exc), "result": None, "stage": stage, "text": text, "context": context,
                "test_cases": _load_test_cases(), "stages": service.known_stages(),
            },
            status_code=400,
        )

    return templates.TemplateResponse(
        request,
        "prompts/run.html",
        {
            "result": result, "stage": stage, "text": text, "context": context, "error": None,
            "test_cases": _load_test_cases(), "stages": service.known_stages(),
        },
    )
