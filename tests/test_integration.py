import json
from pathlib import Path

import pytest
from redisvl.utils.vectorize import HFTextVectorizer

from redis_guardrails import Guardrail, GuardrailService, GuardrailStore

DATA_DIR = Path(__file__).parent.parent / "data"


def _load_guardrails(service: GuardrailService, path: Path) -> None:
    """Test/ops loader: reads a file and calls add_guardrail() per item.
    There is intentionally no bulk load() method in the public API."""
    with open(path) as f:
        raw_guardrails = json.load(f)
    for raw in raw_guardrails:
        service.add_guardrail(Guardrail(**raw))


@pytest.fixture(scope="module")
def service(redis_url):
    vectorizer = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
    store = GuardrailStore(redis_url=redis_url, vectorizer=vectorizer, overwrite=True)
    svc = GuardrailService(store)
    _load_guardrails(svc, DATA_DIR / "guardrails.json")
    return svc


@pytest.mark.integration
def test_seed_guardrails_loaded(service):
    assert len(service.list_guardrails()) == 9


@pytest.mark.integration
def test_testdata_accuracy_meets_bar(service):
    with open(DATA_DIR / "testdata.json") as f:
        cases = json.load(f)

    total = 0
    correct_action = 0
    failures = []

    for case in cases:
        total += 1
        if case["stage"] == "input":
            result = service.evaluate_input(case["input"])
        else:
            result = service.evaluate_output(
                response_text=case["output"], request_text=case["input"]
            )

        if result.action == case["action"]:
            correct_action += 1
        else:
            failures.append((case["id"], case["action"], result.action))

    accuracy = correct_action / total
    if failures:
        print(f"\n{len(failures)}/{total} action mismatches:")
        for case_id, expected, actual in failures:
            print(f"  {case_id}: expected {expected}, got {actual}")

    # Thresholds are untuned placeholders (0.5 everywhere) — this is a
    # provisional bar, not a production accuracy target.
    assert accuracy >= 0.7, f"action accuracy {accuracy:.2f} below 0.7 bar"
