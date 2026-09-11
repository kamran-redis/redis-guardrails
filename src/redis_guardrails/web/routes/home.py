from fastapi import APIRouter, Request

from redis_guardrails.web.templating import templates

router = APIRouter()


@router.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "home.html", {})
