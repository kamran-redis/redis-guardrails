from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import RedirectResponse

from redis_guardrails import GuardrailService
from redis_guardrails.errors import DuplicateGuardrailError, GuardrailNotFoundError, InvalidGuardrailError
from redis_guardrails.models import Action, Guardrail, Stage
from redis_guardrails.web.deps import get_service
from redis_guardrails.web.templating import templates

router = APIRouter()


def _parse_examples(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


@router.get("")
def list_guardrails(
    request: Request,
    stage: Stage | None = Query(default=None),
    flash: str | None = Query(default=None),
    service: GuardrailService = Depends(get_service),
):
    guardrails = service.list_guardrails(stage=stage)
    return templates.TemplateResponse(
        request,
        "guardrails/list.html",
        {"guardrails": guardrails, "stage": stage, "flash": flash},
    )


@router.get("/new")
def new_guardrail_form(request: Request):
    return templates.TemplateResponse(
        request,
        "guardrails/form.html",
        {"mode": "create", "guardrail": None, "error": None},
    )


@router.post("/new")
def create_guardrail(
    request: Request,
    guardrail_id: str = Form(..., alias="id"),
    stage: Stage = Form(...),
    category: str = Form(...),
    description: str = Form(...),
    examples: str = Form(...),
    action: Action = Form(...),
    match_threshold: float = Form(...),
    service: GuardrailService = Depends(get_service),
):
    guardrail = Guardrail(
        id=guardrail_id, stage=stage, category=category, description=description,
        examples=_parse_examples(examples), action=action, match_threshold=match_threshold,
    )
    try:
        service.add_guardrail(guardrail)
    except (InvalidGuardrailError, DuplicateGuardrailError) as exc:
        return templates.TemplateResponse(
            request,
            "guardrails/form.html",
            {"mode": "create", "guardrail": guardrail, "error": str(exc)},
            status_code=400,
        )
    return RedirectResponse(url=f"/guardrails/{guardrail.id}", status_code=303)


@router.get("/{guardrail_id}")
def guardrail_detail(
    request: Request,
    guardrail_id: str,
    service: GuardrailService = Depends(get_service),
):
    guardrail = service.get_guardrail(guardrail_id)
    return templates.TemplateResponse(
        request, "guardrails/detail.html", {"guardrail": guardrail}
    )


@router.get("/{guardrail_id}/edit")
def edit_guardrail_form(
    request: Request,
    guardrail_id: str,
    service: GuardrailService = Depends(get_service),
):
    guardrail = service.get_guardrail(guardrail_id)
    return templates.TemplateResponse(
        request,
        "guardrails/form.html",
        {"mode": "edit", "guardrail": guardrail, "error": None},
    )


@router.post("/{guardrail_id}/edit")
def update_guardrail_route(
    request: Request,
    guardrail_id: str,
    stage: Stage = Form(...),
    category: str = Form(...),
    description: str = Form(...),
    examples: str = Form(...),
    action: Action = Form(...),
    match_threshold: float = Form(...),
    service: GuardrailService = Depends(get_service),
):
    # guardrail_id always comes from the URL path, never the form body —
    # prevents a crafted request from renaming a different guardrail.
    guardrail = Guardrail(
        id=guardrail_id, stage=stage, category=category, description=description,
        examples=_parse_examples(examples), action=action, match_threshold=match_threshold,
    )
    try:
        service.update_guardrail(guardrail)
    except InvalidGuardrailError as exc:
        return templates.TemplateResponse(
            request,
            "guardrails/form.html",
            {"mode": "edit", "guardrail": guardrail, "error": str(exc)},
            status_code=400,
        )
    return RedirectResponse(url=f"/guardrails/{guardrail.id}", status_code=303)


@router.post("/{guardrail_id}/delete")
def delete_guardrail_route(guardrail_id: str, service: GuardrailService = Depends(get_service)):
    try:
        service.delete_guardrail(guardrail_id)
        flash = f"deleted:{guardrail_id}"
    except GuardrailNotFoundError:
        flash = f"not_found:{guardrail_id}"
    return RedirectResponse(url=f"/guardrails?flash={flash}", status_code=303)
