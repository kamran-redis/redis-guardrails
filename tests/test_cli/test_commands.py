import json

from click.testing import CliRunner

from redis_guardrails import GuardrailService
from redis_guardrails.cli.commands import cli
from redis_guardrails.models import Match
from tests.fakes import FakeStore


def _patch_build_service(monkeypatch, service):
    monkeypatch.setattr("redis_guardrails.cli.commands.build_service", lambda **kwargs: service)


def _write_json(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_load_command_success(monkeypatch, tmp_path):
    service = GuardrailService(FakeStore())
    _patch_build_service(monkeypatch, service)

    path = _write_json(tmp_path, "guardrails.json", [
        {"id": "g-1", "stage": "input", "category": "cat", "description": "d",
         "examples": ["ex"], "action": "BLOCK", "match_threshold": 0.5},
    ])

    result = CliRunner().invoke(cli, ["load", str(path)])
    assert result.exit_code == 0
    assert "Loaded 1/1 guardrails." in result.output


def test_load_command_exits_nonzero_on_errors(monkeypatch, tmp_path):
    service = GuardrailService(FakeStore())
    _patch_build_service(monkeypatch, service)

    path = _write_json(tmp_path, "guardrails.json", [{"id": "g-1"}])  # malformed

    result = CliRunner().invoke(cli, ["load", str(path)])
    assert result.exit_code == 1
    assert "Errors (1):" in result.output


def test_benchmark_command_prints_report(monkeypatch, tmp_path):
    store = FakeStore()
    store.matches_by_text["bad text"] = [
        Match(rule_id="g-1", category="cat", action="BLOCK", distance=0.1, threshold=0.5, chunk_id="input-0", evaluated_text="bad text")
    ]
    service = GuardrailService(store)
    _patch_build_service(monkeypatch, service)

    path = _write_json(tmp_path, "testdata.json", [
        {"id": "case-1", "stage": "input", "input": "bad text", "category": "cat", "action": "BLOCK"},
    ])

    result = CliRunner().invoke(cli, ["benchmark", str(path)])
    assert result.exit_code == 0
    assert "case-1" in result.output
    assert "PASS" in result.output


def test_benchmark_command_min_accuracy_gates_exit_code(monkeypatch, tmp_path):
    service = GuardrailService(FakeStore())  # nothing configured -> always ALLOW
    _patch_build_service(monkeypatch, service)

    path = _write_json(tmp_path, "testdata.json", [
        {"id": "case-1", "stage": "input", "input": "anything", "category": "cat", "action": "BLOCK"},
    ])

    result = CliRunner().invoke(cli, ["benchmark", str(path), "--min-accuracy", "0.9"])
    assert result.exit_code == 1


def test_evaluate_input_command_prints_result(monkeypatch):
    service = GuardrailService(FakeStore())
    _patch_build_service(monkeypatch, service)

    result = CliRunner().invoke(cli, ["evaluate", "input", "hello there"])
    assert result.exit_code == 0
    assert "Action: ALLOW" in result.output


def test_evaluate_output_command_requires_response(monkeypatch):
    service = GuardrailService(FakeStore())
    _patch_build_service(monkeypatch, service)

    result = CliRunner().invoke(cli, ["evaluate", "output"])
    assert result.exit_code != 0


def test_evaluate_output_command_with_request_and_trace(monkeypatch):
    store = FakeStore()
    prefixed = "User: what is my balance?\nAssistant: it is obvious"
    store.matches_by_text[prefixed] = [
        Match(rule_id="g-1", category="cat", action="FLAG", distance=0.1, threshold=0.5, chunk_id="output-0", evaluated_text=prefixed)
    ]
    service = GuardrailService(store)
    _patch_build_service(monkeypatch, service)

    result = CliRunner().invoke(
        cli, ["evaluate", "output", "--response", "it is obvious", "--request", "what is my balance?", "--trace"]
    )
    assert result.exit_code == 0
    assert "Action: FLAG" in result.output
    assert "All matches" in result.output
