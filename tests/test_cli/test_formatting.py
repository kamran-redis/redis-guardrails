from redis_guardrails.cli.core import CaseResult, LoadItemError, LoadReport, PerformanceSummary
from redis_guardrails.cli.formatting import (
    format_benchmark_report,
    format_evaluation_result,
    format_load_report,
)
from redis_guardrails.models import Chunk, EvaluationResult, Match, PerformanceInfo


def test_format_load_report_no_errors():
    report = LoadReport(total=2, added=["g-1", "g-2"], errors=[])
    text = format_load_report(report)
    assert "Loaded 2/2 guardrails." in text
    assert "Errors" not in text


def test_format_load_report_with_errors():
    report = LoadReport(
        total=2, added=["g-1"],
        errors=[LoadItemError(index=1, guardrail_id="g-2", error=ValueError("boom"))],
    )
    text = format_load_report(report)
    assert "Loaded 1/2 guardrails." in text
    assert "Errors (1):" in text
    assert "[1] g-2: ValueError - boom" in text


def _eval_result(**overrides) -> EvaluationResult:
    defaults = dict(
        evaluation_id="eval-1", stage="input", status="COMPLETED", action="BLOCK",
        primary_match=None, matches=None, chunks=None,
        performance=PerformanceInfo(embedding_ms=1.0, search_ms=2.0, total_ms=3.0),
    )
    defaults.update(overrides)
    return EvaluationResult(**defaults)


def test_format_evaluation_result_block_with_primary_match():
    match = Match(rule_id="g-1", category="cat", action="BLOCK", distance=0.21, threshold=0.5, chunk_id="input-0", evaluated_text="text")
    result = _eval_result(action="BLOCK", primary_match=match)
    text = format_evaluation_result(result)
    assert "Action: BLOCK" in text
    assert "g-1 (cat)" in text
    assert "0.21" in text


def test_format_evaluation_result_allow_no_match():
    result = _eval_result(action="ALLOW", primary_match=None)
    text = format_evaluation_result(result)
    assert "Action: ALLOW" in text
    assert "no matching guardrail" in text


def test_format_evaluation_result_indeterminate():
    result = _eval_result(
        status="INDETERMINATE", action=None,
        performance=PerformanceInfo(embedding_ms=None, search_ms=None, total_ms=812.4),
    )
    text = format_evaluation_result(result)
    assert "INDETERMINATE" in text
    assert "not allowed" in text
    assert "812.4" in text


def test_format_evaluation_result_trace_shows_matches_and_chunks():
    match = Match(rule_id="g-1", category="cat", action="BLOCK", distance=0.2, threshold=0.5, chunk_id="input-0", evaluated_text="text")
    chunk = Chunk(id="input-0", source="input", start_character=0, end_character=4, text="text", evaluated_text="text")
    result = _eval_result(matches=[match], chunks=[chunk])
    text = format_evaluation_result(result, trace=True)
    assert "All matches (1):" in text
    assert "Chunks (1):" in text
    assert "'text'" in text


def test_format_benchmark_report_includes_case_summary_and_categories():
    case = CaseResult(case_id="c-1", stage="input", category="cat", expected_action="BLOCK", result=_eval_result(action="BLOCK"))
    performance = PerformanceSummary(count=1, avg_embedding_ms=1.0, avg_search_ms=2.0, avg_total_ms=3.0, p95_total_ms=3.0, indeterminate_count=0)
    text = format_benchmark_report([case], performance)
    assert "c-1" in text
    assert "PASS" in text
    assert "Total cases:" in text
    assert "cat" in text
    assert "Performance" in text


def test_format_benchmark_report_handles_case_with_no_category():
    # Seed testdata.json has "safe" cases with category: null (they don't
    # belong to any guardrail category) -- the report must not crash on
    # that, and should render a placeholder instead of the literal None.
    case = CaseResult(case_id="c-1", stage="input", category=None, expected_action="ALLOW", result=_eval_result(action="ALLOW"))
    performance = PerformanceSummary(count=1, avg_embedding_ms=1.0, avg_search_ms=2.0, avg_total_ms=3.0, p95_total_ms=3.0, indeterminate_count=0)
    text = format_benchmark_report([case], performance)
    assert "c-1" in text
    assert "None" not in text
    assert "n/a" in text
