import re
from dataclasses import dataclass
from typing import Literal

from redis_guardrails.errors import InvalidGuardrailError

Scope = str
Action = Literal["ALLOW", "FLAG", "BLOCK"]

_VALID_ACTIONS = {"ALLOW", "FLAG", "BLOCK"}

# Guardrail IDs become RedisVL route names, which get interpolated unescaped
# into an internal RediSearch FILTER expression (see SemanticRouter's
# _distance_threshold_filter). An ID containing a character like an
# apostrophe breaks that filter's syntax -- add_guardrail succeeds (no
# error), but every subsequent evaluate() call on that
# scope then raises internally and gets translated to INDETERMINATE forever
# (route_config is persisted, so it survives a process restart). Restricting
# IDs to a safe character set up front prevents that silent, permanent
# outage. This also naturally rejects empty/whitespace-only IDs (which
# otherwise raise a raw SearchError wrapping a pydantic validation error
# instead of InvalidGuardrailError).
_VALID_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_VALID_SCOPE_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


@dataclass
class Guardrail:
    id: str
    scope: Scope
    category: str
    description: str
    examples: list[str]
    action: Action
    match_threshold: float


def validate_guardrail(guardrail: Guardrail) -> None:
    if not _VALID_ID_PATTERN.match(guardrail.id):
        raise InvalidGuardrailError(
            f"guardrail id {guardrail.id!r} is invalid; must match "
            f"{_VALID_ID_PATTERN.pattern!r} (letters, digits, '.', '_', ':', "
            "'-', 1-128 characters)"
        )
    if not _VALID_SCOPE_PATTERN.match(guardrail.scope):
        raise InvalidGuardrailError(
            f"guardrail {guardrail.id!r} has invalid scope {guardrail.scope!r}; "
            f"must match {_VALID_SCOPE_PATTERN.pattern!r} (letters, digits, '.', '_', ':', "
            "'-', 1-64 characters)"
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
    source: Scope
    start_character: int
    end_character: int
    text: str


@dataclass
class Match:
    rule_id: str
    category: str
    action: Action
    distance: float
    threshold: float
    chunk_id: str
    text: str


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
    scope: Scope
    status: Literal["COMPLETED", "INDETERMINATE"]
    action: Action | None
    primary_match: Match | None
    matches: list[Match] | None
    chunks: list[Chunk] | None
    performance: PerformanceInfo
