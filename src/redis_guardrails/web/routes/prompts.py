from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request

from redis_guardrails import GuardrailService
from redis_guardrails.cli.core import evaluate_prompt_input, evaluate_prompt_output
from redis_guardrails.web.deps import get_service
from redis_guardrails.web.templating import templates

router = APIRouter()


def _parse_optional_int(raw: str) -> int | None:
    raw = raw.strip()
    return int(raw) if raw else None


@router.get("/evaluate")
def evaluate_form(request: Request):
    return templates.TemplateResponse(
        request,
        "prompts/run.html",
        {"result": None, "stage": "input", "text": "", "request_text": "", "error": None},
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
            {"error": str(exc), "result": None, "stage": stage, "text": text, "request_text": request_text},
            status_code=400,
        )

    return templates.TemplateResponse(
        request,
        "prompts/run.html",
        {"result": result, "stage": stage, "text": text, "request_text": request_text, "error": None},
    )
