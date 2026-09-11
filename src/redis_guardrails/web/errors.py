from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from redis.exceptions import RedisError

from redis_guardrails.errors import GuardrailNotFoundError
from redis_guardrails.web.templating import templates


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(GuardrailNotFoundError)
    async def guardrail_not_found_handler(request: Request, exc: GuardrailNotFoundError):
        return templates.TemplateResponse(
            request,
            "error.html",
            {"status_code": 404, "message": f"Guardrail '{exc.guardrail_id}' not found."},
            status_code=404,
        )

    @app.exception_handler(RedisError)
    async def redis_error_handler(request: Request, exc: RedisError):
        return templates.TemplateResponse(
            request,
            "error.html",
            {"status_code": 502, "message": "The guardrails store is unreachable."},
            status_code=502,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return templates.TemplateResponse(
            request,
            "error.html",
            {"status_code": 422, "message": "The submitted form was invalid."},
            status_code=422,
        )
