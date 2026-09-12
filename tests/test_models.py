import pytest

from redis_guardrails.errors import InvalidGuardrailError
from redis_guardrails.models import Guardrail, validate_guardrail


def _guardrail(**overrides) -> Guardrail:
    defaults = dict(
        id="test-001",
        stage="input",
        category="test_category",
        description="a test guardrail",
        examples=["example one"],
        action="BLOCK",
        match_threshold=0.5,
    )
    defaults.update(overrides)
    return Guardrail(**defaults)


def test_valid_guardrail_passes_validation():
    validate_guardrail(_guardrail())  # must not raise


def test_stage_with_unsafe_characters_raises():
    with pytest.raises(InvalidGuardrailError):
        validate_guardrail(_guardrail(stage="bad stage!"))


def test_empty_stage_raises():
    with pytest.raises(InvalidGuardrailError):
        validate_guardrail(_guardrail(stage=""))


def test_novel_stage_name_passes_validation():
    validate_guardrail(_guardrail(stage="input2"))  # must not raise


def test_invalid_action_raises():
    with pytest.raises(InvalidGuardrailError):
        validate_guardrail(_guardrail(action="DENY"))


def test_out_of_range_threshold_raises():
    with pytest.raises(InvalidGuardrailError):
        validate_guardrail(_guardrail(match_threshold=3.0))


def test_zero_examples_raises():
    with pytest.raises(InvalidGuardrailError):
        validate_guardrail(_guardrail(examples=[]))


def test_id_with_apostrophe_raises():
    with pytest.raises(InvalidGuardrailError):
        validate_guardrail(_guardrail(id="bob's-guardrail"))


def test_empty_id_raises():
    with pytest.raises(InvalidGuardrailError):
        validate_guardrail(_guardrail(id=""))


def test_normal_alphanumeric_id_passes():
    validate_guardrail(_guardrail(id="prompt-injection-input-001"))  # must not raise
