import time
import uuid

from redis_guardrails.chunking import chunk_text
from redis_guardrails.errors import (
    EmbeddingError,
    GuardrailNotFoundError,
    IncompleteCoverageError,
    SearchError,
)
from redis_guardrails.evaluator import decide
from redis_guardrails.models import (
    EvaluationResult,
    Guardrail,
    Match,
    PerformanceInfo,
    Stage,
    validate_guardrail,
)
from redis_guardrails.store import GuardrailStore

_INDETERMINATE_ERRORS = (EmbeddingError, SearchError, IncompleteCoverageError)


class GuardrailService:
    def __init__(self, store: GuardrailStore):
        self._store = store

    def add_guardrail(self, guardrail: Guardrail) -> None:
        validate_guardrail(guardrail)
        self._store.add(guardrail)

    def update_guardrail(self, guardrail: Guardrail) -> None:
        validate_guardrail(guardrail)
        self._store.update(guardrail)

    def delete_guardrail(self, guardrail_id: str) -> None:
        self._store.delete(guardrail_id)

    def get_guardrail(self, guardrail_id: str) -> Guardrail:
        guardrail = self._store.get(guardrail_id)
        if guardrail is None:
            raise GuardrailNotFoundError(guardrail_id)
        return guardrail

    def list_guardrails(self, stage: Stage | None = None) -> list[Guardrail]:
        return self._store.list(stage)

    def evaluate_input(self, text: str, include_trace: bool = False) -> EvaluationResult:
        return self._evaluate(stage="input", text=text, prefix="", include_trace=include_trace)

    def evaluate_output(
        self,
        response_text: str,
        request_text: str | None = None,
        include_trace: bool = False,
    ) -> EvaluationResult:
        prefix = f"User: {request_text}\nAssistant: " if request_text is not None else ""
        return self._evaluate(
            stage="output", text=response_text, prefix=prefix, include_trace=include_trace
        )

    def _evaluate(
        self, *, stage: Stage, text: str, prefix: str, include_trace: bool
    ) -> EvaluationResult:
        evaluation_id = f"eval-{uuid.uuid4()}"
        start = time.perf_counter()

        try:
            chunks = chunk_text(text, source=stage, prefix=prefix)

            embedding_start = time.perf_counter()
            vectors = self._store.embed([c.evaluated_text for c in chunks])
            embedding_ms = (time.perf_counter() - embedding_start) * 1000

            search_start = time.perf_counter()
            matches: list[Match] = []
            for chunk, vector in zip(chunks, vectors):
                matches.extend(self._store.search(vector, chunk, stage))
            search_ms = (time.perf_counter() - search_start) * 1000
        except _INDETERMINATE_ERRORS:
            total_ms = (time.perf_counter() - start) * 1000
            return EvaluationResult(
                evaluation_id=evaluation_id,
                stage=stage,
                status="INDETERMINATE",
                action=None,
                primary_match=None,
                matches=None,
                chunks=None,
                performance=PerformanceInfo(embedding_ms=None, search_ms=None, total_ms=total_ms),
            )

        decision = decide(matches)
        total_ms = (time.perf_counter() - start) * 1000

        return EvaluationResult(
            evaluation_id=evaluation_id,
            stage=stage,
            status="COMPLETED",
            action=decision.action,
            primary_match=decision.primary_match,
            matches=decision.matches if include_trace else None,
            chunks=chunks if include_trace else None,
            performance=PerformanceInfo(
                embedding_ms=embedding_ms, search_ms=search_ms, total_ms=total_ms
            ),
        )
