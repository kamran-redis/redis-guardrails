from dataclasses import dataclass
from typing import Literal

from redis_guardrails.errors import InvalidGuardrailError

Stage = Literal["input", "output"]
Action = Literal["ALLOW", "FLAG", "BLOCK"]

_VALID_STAGES = {"input", "output"}
_VALID_ACTIONS = {"ALLOW", "FLAG", "BLOCK"}


@dataclass
class Guardrail:
    id: str
    stage: Stage
    category: str
    description: str
    examples: list[str]
    action: Action
    match_threshold: float


def validate_guardrail(guardrail: Guardrail) -> None:
    if guardrail.stage not in _VALID_STAGES:
        raise InvalidGuardrailError(
            f"guardrail {guardrail.id!r} has invalid stage {guardrail.stage!r}; "
            f"must be one of {sorted(_VALID_STAGES)}"
        )
    if guardrail.action not in _VALID_ACTIONS:
        raise InvalidGuardrailError(
            f"guardrail {guardrail.id!r} has invalid action {guardrail.action!r}; "
            f"must be one of {sorted(_VALID_ACTIONS)}"
        )
    if not (0.0 < guardrail.match_threshold <= 2.0):
        raise InvalidGuardrailError(
            f"guardrail {guardrail.id!r} has invalid match_threshold "
            f"{guardrail.match_threshold!r}; must be in (0.0, 2.0]"
        )
    if not guardrail.examples:
        raise InvalidGuardrailError(
            f"guardrail {guardrail.id!r} must have at least one example"
        )


@dataclass
class Chunk:
    id: str
    source: Stage
    start_character: int
    end_character: int
    text: str
    evaluated_text: str


@dataclass
class Match:
    rule_id: str
    category: str
    action: Action
    distance: float
    threshold: float
    chunk_id: str
    evaluated_text: str


@dataclass
class Decision:
    action: Action
    primary_match: Match | None
    matches: list[Match]


@dataclass
class PerformanceInfo:
    embedding_ms: float | None
    search_ms: float | None
    total_ms: float


@dataclass
class EvaluationResult:
    evaluation_id: str
    stage: Stage
    status: Literal["COMPLETED", "INDETERMINATE"]
    action: Action | None
    primary_match: Match | None
    matches: list[Match] | None
    chunks: list[Chunk] | None
    performance: PerformanceInfo
