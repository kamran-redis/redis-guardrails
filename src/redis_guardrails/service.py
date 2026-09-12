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
    Scope,
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

    def list_guardrails(self, scope: Scope | None = None) -> list[Guardrail]:
        return self._store.list(scope)

    def known_scopes(self) -> list[str]:
        return self._store.known_scopes()

    def evaluate(
        self,
        scope: str,
        text: str,
        include_trace: bool = False,
        max_chars: int | None = None,
        overlap_chars: int | None = None,
        max_chunks: int | None = None,
    ) -> EvaluationResult:
        evaluation_id = f"eval-{uuid.uuid4()}"
        start = time.perf_counter()

        # Only override chunk_text's own defaults for parameters the caller
        # actually specified — passing None through would mean "chunk into
        # windows of size None", not "use the default".
        chunk_kwargs = {}
        if max_chars is not None:
            chunk_kwargs["max_chars"] = max_chars
        if overlap_chars is not None:
            chunk_kwargs["overlap_chars"] = overlap_chars
        if max_chunks is not None:
            chunk_kwargs["max_chunks"] = max_chunks

        try:
            chunks = chunk_text(text, source=scope, **chunk_kwargs)

            embedding_start = time.perf_counter()
            vectors = self._store.embed([c.text for c in chunks])
            embedding_ms = (time.perf_counter() - embedding_start) * 1000

            search_start = time.perf_counter()
            matches: list[Match] = []
            for chunk, vector in zip(chunks, vectors):
                matches.extend(self._store.search(vector, chunk, scope))
            search_ms = (time.perf_counter() - search_start) * 1000
        except _INDETERMINATE_ERRORS:
            total_ms = (time.perf_counter() - start) * 1000
            return EvaluationResult(
                evaluation_id=evaluation_id,
                scope=scope,
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
            scope=scope,
            status="COMPLETED",
            action=decision.action,
            primary_match=decision.primary_match,
            matches=decision.matches if include_trace else None,
            chunks=chunks if include_trace else None,
            performance=PerformanceInfo(
                embedding_ms=embedding_ms, search_ms=search_ms, total_ms=total_ms
            ),
        )
