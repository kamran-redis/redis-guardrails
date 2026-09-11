from fastapi import Request

from redis_guardrails import GuardrailService


def get_service(request: Request) -> GuardrailService:
    return request.app.state.service
