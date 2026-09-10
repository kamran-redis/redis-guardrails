import hashlib
import struct

import pytest
from redisvl.extensions.router import Route, RoutingConfig, SemanticRouter
from redisvl.extensions.router.schema import DistanceAggregationMethod
from redisvl.utils.vectorize.base import BaseVectorizer


class HashVectorizer(BaseVectorizer):
    """Deterministic 8-dim fake embedding — no model download, no network."""

    def __init__(self, dims: int = 8, **kwargs):
        super().__init__(model="hash-fake", dims=dims, **kwargs)

    def _hash_to_vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return list(struct.unpack("8f", digest[:32]))

    # CORRECTED: BaseVectorizer.embed()/embed_many() are template methods —
    # they call _process_embedding(), which honors `as_buffer=True` by
    # converting the raw float list to a binary buffer via array_to_buffer().
    # SemanticRouter._add_routes() always calls vectorizer.embed_many(...,
    # as_buffer=True) internally, expecting bytes back. Overriding the
    # public embed()/embed_many() directly (as originally written here)
    # bypasses that conversion and raises
    # "Hash field 'vector' has unsupported value type list" on load. The
    # real extension points are the private hooks _embed()/_embed_many().
    def _embed(self, text: str = "", content: str = "", **kwargs) -> list[float]:
        return self._hash_to_vector(content or text)

    def _embed_many(
        self, texts: list[str] = None, contents: list[str] = None, **kwargs
    ) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in (contents or texts)]


@pytest.mark.integration
def test_route_references_and_route_many_shapes(redis_url):
    vectorizer = HashVectorizer()
    route = Route(
        name="contract-test-route",
        references=["alpha reference", "beta reference"],
        metadata={"category": "test", "action": "BLOCK", "description": "d"},
        distance_threshold=0.5,
    )
    router = SemanticRouter(
        name="contract-test-router",
        routes=[route],
        vectorizer=vectorizer,
        routing_config=RoutingConfig(max_k=10, aggregation_method=DistanceAggregationMethod.min),
        redis_url=redis_url,
        overwrite=True,
    )

    fetched = router.get("contract-test-route")
    assert fetched is not None
    assert fetched.name == "contract-test-route"
    assert fetched.metadata["category"] == "test"
    assert fetched.distance_threshold == 0.5

    references = router.get_route_references(route_name="contract-test-route")
    assert len(references) == 2
    reference_texts = {ref["reference"] for ref in references}
    assert reference_texts == {"alpha reference", "beta reference"}

    query_vector = vectorizer.embed("alpha reference")
    route_matches = router.route_many(
        vector=query_vector,
        max_k=10,
        aggregation_method=DistanceAggregationMethod.min,
        distance_threshold=None,
    )
    assert len(route_matches) >= 1
    best = route_matches[0]
    assert best.name == "contract-test-route"
    assert best.distance is not None

    assert isinstance(router.routes, list)
    assert any(r.name == "contract-test-route" for r in router.routes)

    # CORRECTED: `from_existing()` does NOT accept an explicit `vectorizer=`
    # kwarg to override reconstruction. Any unrecognized kwarg (anything
    # other than `validate_on_load`, `lib_name`, `connection_kwargs`, or a
    # leading-underscore key) is forwarded straight to
    # RedisConnectionFactory.get_redis_connection() -> the redis-py
    # Connection constructor, which raises TypeError on an unknown kwarg
    # like `vectorizer`.
    with pytest.raises(TypeError):
        SemanticRouter.from_existing(
            name="contract-test-router", redis_url=redis_url, vectorizer=vectorizer
        )

    # CORRECTED: `from_existing()` always reconstructs the vectorizer from
    # the router's stored config via `vectorizer_from_dict()`, which only
    # recognizes RedisVL's built-in vectorizer types (openai, cohere, hf,
    # mistral, ollama, vertexai, google_genai, voyageai, azure_openai) —
    # keyed off `BaseVectorizer.type`. A custom/local vectorizer subclass
    # (like this test's HashVectorizer, whose `.type` is the inherited
    # "base") cannot be reconstructed and raises ValueError. Concretely:
    # `from_existing()` can only be used to reattach to routers that were
    # originally built with one of RedisVL's built-in vectorizer classes.
    # Task 4 must account for this constraint wherever it reattaches to a
    # router (e.g. it cannot use a raw hash/test vectorizer in production
    # and expect `from_existing()` to work).
    with pytest.raises(ValueError, match="Unable to load vectorizer"):
        SemanticRouter.from_existing(name="contract-test-router", redis_url=redis_url)

    router.remove_route("contract-test-route")
    assert router.get("contract-test-route") is None


@pytest.mark.integration
def test_from_existing_raises_when_index_missing(redis_url):
    with pytest.raises(Exception):
        SemanticRouter.from_existing(name="no-such-router-index", redis_url=redis_url)
