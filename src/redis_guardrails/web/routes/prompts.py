from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request

from redis_guardrails import GuardrailService
from redis_guardrails.cli.core import evaluate_prompt
from redis_guardrails.web.deps import get_service
from redis_guardrails.web.templating import templates

router = APIRouter()


def _parse_optional_int(raw: str) -> int | None:
    raw = raw.strip()
    return int(raw) if raw else None


@router.get("/evaluate")
def evaluate_form(request: Request, service: GuardrailService = Depends(get_service)):
    return templates.TemplateResponse(
        request,
        "prompts/run.html",
        {
            "result": None, "scope": "", "text": "", "error": None,
            "scopes": service.known_scopes(),
        },
    )


@router.post("/evaluate")
def evaluate_submit(
    request: Request,
    scope: str = Form(...),
    text: str = Form(...),
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
        result = evaluate_prompt(service, scope, text, **overrides)
    except ValueError as exc:
        return templates.TemplateResponse(
            request,
            "prompts/run.html",
            {
                "error": str(exc), "result": None, "scope": scope, "text": text,
                "scopes": service.known_scopes(),
            },
            status_code=400,
        )

    return templates.TemplateResponse(
        request,
        "prompts/run.html",
        {
            "result": result, "scope": scope, "text": text, "error": None,
            "scopes": service.known_scopes(),
        },
    )
