from __future__ import annotations

from redis_guardrails.cli.core import CaseResult, LoadReport, PerformanceSummary, classify, summarize_outcomes
from redis_guardrails.models import EvaluationResult


def format_load_report(report: LoadReport) -> str:
    lines = [f"Loaded {len(report.added)}/{report.total} guardrails."]
    if report.errors:
        lines.append("")
        lines.append(f"Errors ({len(report.errors)}):")
        for item in report.errors:
            id_display = item.guardrail_id or "<unknown id>"
            lines.append(f"  [{item.index}] {id_display}: {type(item.error).__name__} - {item.error}")
    return "\n".join(lines)


def format_benchmark_report(cases: list[CaseResult], performance: PerformanceSummary) -> str:
    lines: list[str] = []

    header = f"{'Case':<32} {'Scope':<7} {'Category':<21} {'Expected':<9} {'Actual':<7} Result"
    lines.append(header)
    lines.append("-" * len(header))
    for case in cases:
        actual = case.result.action if case.result.action is not None else "n/a"
        category_display = case.category if case.category is not None else "n/a"
        lines.append(
            f"{case.case_id:<32} {case.scope:<7} {category_display:<21} "
            f"{case.expected_action:<9} {actual:<7} {classify(case)}"
        )

    outcomes = summarize_outcomes(cases)

    lines.append("")
    lines.append("Summary")
    lines.append("-" * len("Summary"))
    lines.append(f"Total cases:     {outcomes.total:>4}")
    lines.append(f"Passed:          {outcomes.passed:>4}  ({outcomes.accuracy:.1f}%)")
    lines.append(f"False positives: {outcomes.false_positives:>4}")
    lines.append(f"False negatives: {outcomes.false_negatives:>4}")
    lines.append(f"Wrong severity:  {outcomes.wrong_severity:>4}")
    lines.append(f"Indeterminate:   {outcomes.indeterminate:>4}")

    lines.append("")
    lines.append("Accuracy by category")
    lines.append("-" * len("Accuracy by category"))
    lines.append(f"{'Category':<20} {'Total':>6} {'Passed':>7} {'Accuracy':>9}")
    for cat in outcomes.by_category:
        lines.append(f"{cat.category:<20} {cat.total:>6} {cat.passed:>7} {cat.accuracy:>8.1f}%")

    lines.append("")
    lines.append("Performance")
    lines.append("-----------")
    lines.append(f"Evaluations:        {performance.count:>4}")
    embedding_display = f"{performance.avg_embedding_ms:.1f} ms" if performance.avg_embedding_ms is not None else "n/a"
    search_display = f"{performance.avg_search_ms:.1f} ms" if performance.avg_search_ms is not None else "n/a"
    lines.append(f"Avg embedding time: {embedding_display}")
    lines.append(f"Avg search time:    {search_display}")
    lines.append(f"Avg total time:     {performance.avg_total_ms:.1f} ms")
    lines.append(f"p95 total time:     {performance.p95_total_ms:.1f} ms")

    return "\n".join(lines)


def format_evaluation_result(result: EvaluationResult, trace: bool = False) -> str:
    lines = [f"Evaluation {result.evaluation_id} ({result.scope}) -> {result.status}"]

    if result.status == "INDETERMINATE":
        lines.append("Action: n/a — evaluation could not complete safely; treat as not allowed")
        lines.append(f"Performance: total {result.performance.total_ms:.1f}ms (embedding/search timing unavailable)")
        return "\n".join(lines)

    lines.append(f"Action: {result.action}")
    if result.primary_match is not None:
        m = result.primary_match
        lines.append(
            f"Primary match: {m.rule_id} ({m.category}), distance {m.distance:.2f} <= threshold {m.threshold:.2f}"
        )
    else:
        lines.append("Primary match: none (no matching guardrail)")

    if result.matches is not None:
        lines.append("")
        lines.append(f"All matches ({len(result.matches)}):")
        for m in result.matches:
            lines.append(
                f"  {m.rule_id:<30} {m.category:<20} {m.action:<5} "
                f"distance={m.distance:.2f} threshold={m.threshold:.2f}  chunk={m.chunk_id}"
            )
            lines.append(f"    text: {m.text!r}")

    if trace and result.chunks is not None:
        lines.append("")
        lines.append(f"Chunks ({len(result.chunks)}):")
        for c in result.chunks:
            lines.append(f"  [{c.id}] chars {c.start_character}-{c.end_character}")
            lines.append(f"    text: {c.text!r}")
            chunk_matches = [m for m in (result.matches or []) if m.chunk_id == c.id]
            if chunk_matches:
                lines.append(f"    matches ({len(chunk_matches)}):")
                for m in chunk_matches:
                    lines.append(
                        f"      {m.rule_id:<30} {m.category:<20} {m.action:<5} "
                        f"distance={m.distance:.2f} threshold={m.threshold:.2f}"
                    )
            else:
                lines.append("    matches: none")

    lines.append("")
    p = result.performance
    lines.append(f"Performance: embedding {p.embedding_ms:.1f}ms, search {p.search_ms:.1f}ms, total {p.total_ms:.1f}ms")

    return "\n".join(lines)
