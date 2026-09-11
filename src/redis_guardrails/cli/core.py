from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from redisvl.utils.vectorize import HFTextVectorizer

from redis_guardrails import Guardrail, GuardrailService, GuardrailStore
from redis_guardrails.errors import GuardrailError
from redis_guardrails.models import Action, EvaluationResult, Stage

DEFAULT_REDIS_URL = "redis://localhost:6379"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_service(
    redis_url: str = DEFAULT_REDIS_URL,
    model: str = DEFAULT_MODEL,
    overwrite: bool = False,
) -> GuardrailService:
    try:
        vectorizer = HFTextVectorizer(model=model)
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required to run this command. "
            "Install with `pip install -e '.[cli]'`."
        ) from exc
    store = GuardrailStore(redis_url=redis_url, vectorizer=vectorizer, overwrite=overwrite)
    return GuardrailService(store)


@dataclass
class LoadItemError:
    index: int
    guardrail_id: str | None
    error: Exception


@dataclass
class LoadReport:
    total: int
    added: list[str]
    errors: list[LoadItemError]


def load_guardrails_from_file(service: GuardrailService, path: Path) -> LoadReport:
    with open(path) as f:
        raw_guardrails = json.load(f)

    added: list[str] = []
    errors: list[LoadItemError] = []
    for index, raw in enumerate(raw_guardrails):
        guardrail_id = raw.get("id") if isinstance(raw, dict) else None
        try:
            guardrail = Guardrail(**raw)
            service.add_guardrail(guardrail)
            added.append(guardrail.id)
        except (GuardrailError, TypeError) as exc:
            errors.append(LoadItemError(index=index, guardrail_id=guardrail_id, error=exc))

    return LoadReport(total=len(raw_guardrails), added=added, errors=errors)


def evaluate_prompt_input(
    service: GuardrailService, text: str, trace: bool = False
) -> EvaluationResult:
    return service.evaluate_input(text, include_trace=trace)


def evaluate_prompt_output(
    service: GuardrailService,
    response_text: str,
    request_text: str | None = None,
    trace: bool = False,
) -> EvaluationResult:
    return service.evaluate_output(
        response_text=response_text, request_text=request_text, include_trace=trace
    )


@dataclass
class CaseResult:
    case_id: str
    stage: Stage
    category: str | None
    expected_action: Action
    result: EvaluationResult


def run_benchmark(service: GuardrailService, path: Path) -> list[CaseResult]:
    with open(path) as f:
        cases = json.load(f)

    results: list[CaseResult] = []
    for case in cases:
        if case["stage"] == "input":
            result = service.evaluate_input(case["input"], include_trace=True)
        else:
            result = service.evaluate_output(
                response_text=case["output"], request_text=case.get("input"), include_trace=True
            )
        results.append(
            CaseResult(
                case_id=case["id"],
                stage=case["stage"],
                category=case["category"],
                expected_action=case["action"],
                result=result,
            )
        )
    return results


def classify(case: CaseResult) -> str:
    if case.result.status == "INDETERMINATE":
        return "INDETERMINATE"
    if case.result.action == case.expected_action:
        return "PASS"
    if case.expected_action == "ALLOW":
        return "FALSE_POSITIVE"
    if case.result.action == "ALLOW":
        return "FALSE_NEGATIVE"
    return "WRONG_SEVERITY"


@dataclass
class PerformanceSummary:
    count: int
    avg_embedding_ms: float | None
    avg_search_ms: float | None
    avg_total_ms: float
    p95_total_ms: float
    indeterminate_count: int


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def summarize_performance(cases: list[CaseResult]) -> PerformanceSummary:
    performances = [c.result.performance for c in cases]
    totals = [p.total_ms for p in performances]
    embeddings = [p.embedding_ms for p in performances if p.embedding_ms is not None]
    searches = [p.search_ms for p in performances if p.search_ms is not None]
    indeterminate_count = sum(1 for c in cases if c.result.status == "INDETERMINATE")

    return PerformanceSummary(
        count=len(cases),
        avg_embedding_ms=(sum(embeddings) / len(embeddings)) if embeddings else None,
        avg_search_ms=(sum(searches) / len(searches)) if searches else None,
        avg_total_ms=(sum(totals) / len(totals)) if totals else 0.0,
        p95_total_ms=_percentile(sorted(totals), 0.95),
        indeterminate_count=indeterminate_count,
    )
