import json
from pathlib import Path

import pytest

from redis_guardrails import GuardrailService
from redis_guardrails.cli.core import (
    build_service,
    evaluate_prompt_input,
    evaluate_prompt_output,
    load_guardrails_from_file,
)
from redis_guardrails.errors import DuplicateGuardrailError
from tests.fakes import FakeStore


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def service(store):
    return GuardrailService(store)


def _guardrail_dict(**overrides) -> dict:
    defaults = dict(
        id="g-1", stage="input", category="cat", description="d",
        examples=["ex"], action="BLOCK", match_threshold=0.5,
    )
    defaults.update(overrides)
    return defaults


def _write_json(tmp_path: Path, name: str, data) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_load_all_valid_guardrails(service, tmp_path):
    path = _write_json(tmp_path, "guardrails.json", [_guardrail_dict(id="g-1"), _guardrail_dict(id="g-2")])
    report = load_guardrails_from_file(service, path)
    assert report.total == 2
    assert report.added == ["g-1", "g-2"]
    assert report.errors == []
    assert {g.id for g in service.list_guardrails()} == {"g-1", "g-2"}


def test_load_collects_duplicate_error_and_continues(service, tmp_path):
    path = _write_json(
        tmp_path, "guardrails.json",
        [_guardrail_dict(id="g-1"), _guardrail_dict(id="g-1"), _guardrail_dict(id="g-2")],
    )
    report = load_guardrails_from_file(service, path)
    assert report.total == 3
    assert report.added == ["g-1", "g-2"]
    assert len(report.errors) == 1
    assert report.errors[0].index == 1
    assert report.errors[0].guardrail_id == "g-1"
    assert isinstance(report.errors[0].error, DuplicateGuardrailError)


def test_load_collects_invalid_guardrail_error_and_continues(service, tmp_path):
    path = _write_json(
        tmp_path, "guardrails.json",
        [_guardrail_dict(id="g-1", action="NOT_A_REAL_ACTION"), _guardrail_dict(id="g-2")],
    )
    report = load_guardrails_from_file(service, path)
    assert report.added == ["g-2"]
    assert len(report.errors) == 1
    assert report.errors[0].guardrail_id == "g-1"


def test_load_collects_malformed_record_as_type_error(service, tmp_path):
    path = _write_json(tmp_path, "guardrails.json", [{"id": "g-1"}])  # missing required fields
    report = load_guardrails_from_file(service, path)
    assert report.added == []
    assert len(report.errors) == 1
    assert report.errors[0].guardrail_id == "g-1"


def test_evaluate_prompt_input_delegates_to_service(service, store):
    from redis_guardrails.models import Match

    store.matches_by_text["hello"] = [
        Match(rule_id="g-1", category="cat", action="BLOCK", distance=0.1, threshold=0.5, chunk_id="input-0", evaluated_text="hello")
    ]
    result = evaluate_prompt_input(service, "hello")
    assert result.action == "BLOCK"


def test_evaluate_prompt_input_trace_flag_populates_matches(service, store):
    from redis_guardrails.models import Match

    store.matches_by_text["hello"] = [
        Match(rule_id="g-1", category="cat", action="BLOCK", distance=0.1, threshold=0.5, chunk_id="input-0", evaluated_text="hello")
    ]
    result = evaluate_prompt_input(service, "hello", trace=True)
    assert result.matches is not None


def test_evaluate_prompt_output_passes_request_text_through(service, store):
    from redis_guardrails.models import Match

    prefixed = "User: what is my balance?\nAssistant: it is obvious"
    store.matches_by_text[prefixed] = [
        Match(rule_id="g-1", category="cat", action="FLAG", distance=0.1, threshold=0.5, chunk_id="output-0", evaluated_text=prefixed)
    ]
    result = evaluate_prompt_output(service, "it is obvious", request_text="what is my balance?")
    assert result.action == "FLAG"
