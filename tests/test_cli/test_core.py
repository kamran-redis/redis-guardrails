import json
from pathlib import Path

import pytest

from redis_guardrails import GuardrailService
from redis_guardrails.cli.core import (
    CaseResult,
    PerformanceSummary,
    build_service,
    classify,
    evaluate_prompt_input,
    evaluate_prompt_output,
    load_guardrails_from_file,
    run_benchmark,
    summarize_performance,
)
from redis_guardrails.errors import DuplicateGuardrailError
from redis_guardrails.models import EvaluationResult, Match, PerformanceInfo
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
    store.matches_by_text["hello"] = [
        Match(rule_id="g-1", category="cat", action="BLOCK", distance=0.1, threshold=0.5, chunk_id="input-0", evaluated_text="hello")
    ]
    result = evaluate_prompt_input(service, "hello")
    assert result.action == "BLOCK"


def test_evaluate_prompt_input_trace_flag_populates_matches(service, store):
    store.matches_by_text["hello"] = [
        Match(rule_id="g-1", category="cat", action="BLOCK", distance=0.1, threshold=0.5, chunk_id="input-0", evaluated_text="hello")
    ]
    result = evaluate_prompt_input(service, "hello", trace=True)
    assert result.matches is not None


def test_evaluate_prompt_output_passes_request_text_through(service, store):
    prefixed = "User: what is my balance?\nAssistant: it is obvious"
    store.matches_by_text[prefixed] = [
        Match(rule_id="g-1", category="cat", action="FLAG", distance=0.1, threshold=0.5, chunk_id="output-0", evaluated_text=prefixed)
    ]
    result = evaluate_prompt_output(service, "it is obvious", request_text="what is my balance?")
    assert result.action == "FLAG"


def _eval_result(**overrides) -> EvaluationResult:
    defaults = dict(
        evaluation_id="eval-1", stage="input", status="COMPLETED", action="BLOCK",
        primary_match=None, matches=None, chunks=None,
        performance=PerformanceInfo(embedding_ms=1.0, search_ms=2.0, total_ms=3.0),
    )
    defaults.update(overrides)
    return EvaluationResult(**defaults)


def _case_result(**overrides) -> CaseResult:
    defaults = dict(
        case_id="c-1", stage="input", category="cat", expected_action="BLOCK", result=_eval_result()
    )
    defaults.update(overrides)
    return CaseResult(**defaults)


def test_classify_pass():
    case = _case_result(expected_action="BLOCK", result=_eval_result(action="BLOCK"))
    assert classify(case) == "PASS"


def test_classify_false_positive():
    case = _case_result(expected_action="ALLOW", result=_eval_result(action="BLOCK"))
    assert classify(case) == "FALSE_POSITIVE"


def test_classify_false_negative():
    case = _case_result(expected_action="BLOCK", result=_eval_result(action="ALLOW"))
    assert classify(case) == "FALSE_NEGATIVE"


def test_classify_wrong_severity():
    case = _case_result(expected_action="BLOCK", result=_eval_result(action="FLAG"))
    assert classify(case) == "WRONG_SEVERITY"


def test_classify_indeterminate():
    case = _case_result(result=_eval_result(status="INDETERMINATE", action=None))
    assert classify(case) == "INDETERMINATE"


def test_summarize_performance_averages_and_skips_none():
    cases = [
        _case_result(
            result=_eval_result(
                performance=PerformanceInfo(embedding_ms=2.0, search_ms=4.0, total_ms=10.0)
            )
        ),
        _case_result(
            result=_eval_result(
                status="INDETERMINATE", action=None,
                performance=PerformanceInfo(embedding_ms=None, search_ms=None, total_ms=20.0),
            )
        ),
    ]
    summary = summarize_performance(cases)
    assert summary.count == 2
    assert summary.avg_embedding_ms == 2.0
    assert summary.avg_search_ms == 4.0
    assert summary.avg_total_ms == 15.0
    assert summary.indeterminate_count == 1


def test_summarize_performance_all_indeterminate_gives_none_embedding_and_search():
    cases = [
        _case_result(
            result=_eval_result(
                status="INDETERMINATE", action=None,
                performance=PerformanceInfo(embedding_ms=None, search_ms=None, total_ms=5.0),
            )
        ),
    ]
    summary = summarize_performance(cases)
    assert summary.avg_embedding_ms is None
    assert summary.avg_search_ms is None
    assert summary.avg_total_ms == 5.0


def test_summarize_performance_p95():
    cases = [
        _case_result(
            result=_eval_result(performance=PerformanceInfo(embedding_ms=1.0, search_ms=1.0, total_ms=float(v)))
        )
        for v in [10, 20, 30, 40, 50]
    ]
    summary = summarize_performance(cases)
    assert summary.p95_total_ms == 50.0


def test_run_benchmark_evaluates_input_and_output_cases(tmp_path):
    store = FakeStore()
    service = GuardrailService(store)
    store.matches_by_text["bad text"] = [
        Match(rule_id="g-1", category="cat", action="BLOCK", distance=0.1, threshold=0.5, chunk_id="input-0", evaluated_text="bad text")
    ]

    path = _write_json(tmp_path, "testdata.json", [
        {"id": "case-1", "stage": "input", "input": "bad text", "category": "cat", "action": "BLOCK"},
        {"id": "case-2", "stage": "output", "input": "req", "output": "resp", "category": "cat", "action": "ALLOW"},
    ])

    results = run_benchmark(service, path)
    assert len(results) == 2
    assert results[0].case_id == "case-1"
    assert results[0].result.action == "BLOCK"
    assert results[1].case_id == "case-2"
    assert results[1].result.action == "ALLOW"


def test_build_service_raises_runtime_error_when_sentence_transformers_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    with pytest.raises(RuntimeError, match="sentence-transformers"):
        build_service()
