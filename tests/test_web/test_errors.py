from fastapi import APIRouter
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from redis.exceptions import RedisError

from redis_guardrails.errors import GuardrailNotFoundError
from redis_guardrails.web.app import _build_app


def _client_with_throwaway_route(exc: Exception):
    app = _build_app()
    router = APIRouter()

    @router.get("/__raise")
    def raise_it():
        raise exc

    app.include_router(router)
    return TestClient(app)


def test_guardrail_not_found_renders_404_page():
    client = _client_with_throwaway_route(GuardrailNotFoundError("g-1"))
    response = client.get("/__raise")
    assert response.status_code == 404
    assert "g-1" in response.text
    assert "not found" in response.text


def test_redis_error_renders_502_page():
    client = _client_with_throwaway_route(RedisError("connection refused"))
    response = client.get("/__raise")
    assert response.status_code == 502
    assert "unreachable" in response.text


def test_request_validation_error_renders_422_page():
    client = _client_with_throwaway_route(RequestValidationError(errors=[]))
    response = client.get("/__raise")
    assert response.status_code == 422
    assert "invalid" in response.text
