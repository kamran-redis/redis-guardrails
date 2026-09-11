import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from redis_guardrails import GuardrailService
from redis_guardrails.web.app import _build_app
from redis_guardrails.web.deps import get_service
from tests.fakes import FakeStore


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def service(store):
    return GuardrailService(store)


@pytest.fixture
def client(service):
    app = _build_app()
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)
