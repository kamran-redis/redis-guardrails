import json
from pathlib import Path

import pytest
from redisvl.utils.vectorize import HFTextVectorizer

from redis_guardrails import Guardrail, GuardrailService, GuardrailStore
from redis_guardrails.errors import IncompleteCoverageError

DATA_DIR = Path(__file__).parent.parent / "data"


def _load_guardrails(service: GuardrailService, path: Path) -> None:
    """Test/ops loader: reads a file and calls add_guardrail() per item.
    There is intentionally no bulk load() method in the public API."""
    with open(path) as f:
        raw_guardrails = json.load(f)
    for raw in raw_guardrails:
        service.add_guardrail(Guardrail(**raw))


@pytest.fixture(scope="module")
def service(redis_url, allow_test_overwrite):
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


@pytest.fixture(scope="module")
def hf_store(redis_url):
    """A store using the REAL HFTextVectorizer, not the fake HashVectorizer.

    Uses overwrite=False (attach, don't wipe) so this can't clobber the
    `service` fixture's seeded guardrails regardless of test execution
    order -- this fixture never adds/removes routes, it's only used to
    exercise embed()'s token-limit-awareness for this vectorizer type.
    """
    vectorizer = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
    return GuardrailStore(redis_url=redis_url, vectorizer=vectorizer, overwrite=False)


@pytest.mark.integration
def test_embed_raises_on_text_that_would_be_silently_truncated(hf_store):
    # all-MiniLM-L6-v2's max_seq_length is 256 tokens. This is well within
    # chunking.py's DEFAULT_MAX_CHARS=800 character budget (540 chars) but
    # tokenizes to ~542 tokens because CJK text is far more token-dense per
    # character than English -- exactly the "within char budget but over
    # token budget" case the reviewer flagged as silently truncated today.
    token_dense_text = "这是一个测试句子。" * 60
    with pytest.raises(IncompleteCoverageError):
        hf_store.embed([token_dense_text])


@pytest.mark.integration
def test_embed_normal_text_still_works_with_hf_vectorizer(hf_store):
    vectors = hf_store.embed(["A perfectly normal, short piece of text."])
    assert len(vectors) == 1
    assert len(vectors[0]) > 0
