from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request

from redis_guardrails import GuardrailService
from redis_guardrails.cli.core import evaluate_prompt_input, evaluate_prompt_output
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
                if stage == "output":
                    text, request_text = case["output"], case.get("input", "")
                else:
                    text, request_text = case["input"], ""
                case_id = case["id"]
            except KeyError:
                continue
            cases.append({
                "file": path.name,
                "id": case_id,
                "stage": stage,
                "category": case.get("category"),
                "text": text,
                "request_text": request_text,
            })
    return cases


@router.get("/evaluate")
def evaluate_form(request: Request):
    return templates.TemplateResponse(
        request,
        "prompts/run.html",
        {
            "result": None, "stage": "input", "text": "", "request_text": "", "error": None,
            "test_cases": _load_test_cases(),
        },
    )


@router.post("/evaluate")
def evaluate_submit(
    request: Request,
    stage: str = Form(...),
    text: str = Form(...),
    request_text: str = Form(default=""),
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
        if stage == "output":
            result = evaluate_prompt_output(service, text, request_text=request_text or None, **overrides)
        else:
            result = evaluate_prompt_input(service, text, **overrides)
    except ValueError as exc:
        return templates.TemplateResponse(
            request,
            "prompts/run.html",
            {
                "error": str(exc), "result": None, "stage": stage, "text": text, "request_text": request_text,
                "test_cases": _load_test_cases(),
            },
            status_code=400,
        )

    return templates.TemplateResponse(
        request,
        "prompts/run.html",
        {
            "result": result, "stage": stage, "text": text, "request_text": request_text, "error": None,
            "test_cases": _load_test_cases(),
        },
    )
