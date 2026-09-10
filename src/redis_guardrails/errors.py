class GuardrailError(Exception):
    """Base class for all redis_guardrails errors."""


class GuardrailNotFoundError(GuardrailError):
    def __init__(self, guardrail_id: str):
        super().__init__(f"no guardrail with id {guardrail_id!r}")
        self.guardrail_id = guardrail_id


class DuplicateGuardrailError(GuardrailError):
    def __init__(self, guardrail_id: str):
        super().__init__(f"a guardrail with id {guardrail_id!r} already exists")
        self.guardrail_id = guardrail_id


class InvalidGuardrailError(GuardrailError):
    pass


class EmbeddingError(GuardrailError):
    pass


class SearchError(GuardrailError):
    pass


class IncompleteCoverageError(GuardrailError):
    pass
