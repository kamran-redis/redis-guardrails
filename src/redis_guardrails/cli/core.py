from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from redisvl.utils.vectorize import HFTextVectorizer

from redis_guardrails import Guardrail, GuardrailService, GuardrailStore
from redis_guardrails.errors import GuardrailError
from redis_guardrails.models import EvaluationResult

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
