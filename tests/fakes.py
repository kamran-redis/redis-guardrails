from __future__ import annotations

import hashlib
import struct

from redisvl.utils.vectorize.base import BaseVectorizer


class HashVectorizer(BaseVectorizer):
    """Deterministic 8-dim fake embedding — no model download, no network.

    Same text always hashes to the same vector, so tests can assert on
    which guardrail matches without needing a real embedding model.
    """

    def __init__(self, dims: int = 8, **kwargs):
        super().__init__(model="hash-fake", dims=dims, **kwargs)

    def _hash_to_vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return list(struct.unpack("8f", digest[:32]))

    # Override the private hooks, not the public embed()/embed_many() —
    # BaseVectorizer's public methods are template methods that funnel
    # through _process_embedding() (which honors as_buffer=True, converting
    # to the binary buffer SemanticRouter._add_routes() expects). Overriding
    # the public methods directly bypasses that and breaks hash storage.
    # Confirmed against redisvl 0.27.2 in Task 1's contract test.
    def _embed(self, text: str = "", content: str = "", **kwargs) -> list[float]:
        return self._hash_to_vector(content or text)

    def _embed_many(
        self, texts: list[str] = None, contents: list[str] = None, **kwargs
    ) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in (contents or texts)]


from redis_guardrails.errors import GuardrailNotFoundError
from redis_guardrails.models import Chunk, Guardrail, Match


class FakeStore:
    """In-memory stand-in for GuardrailStore — no Redis, no embeddings.

    Tests configure `.matches_by_text` and `.raise_on_search`/`.raise_on_embed`
    to control what evaluate_input/evaluate_output see, without needing a
    real vector search.
    """

    def __init__(self):
        self._guardrails: dict[str, Guardrail] = {}
        self.matches_by_text: dict[str, list[Match]] = {}
        self.raise_on_embed: Exception | None = None
        self.raise_on_search: Exception | None = None
        self.embedded_texts: list[str] = []

    def add(self, guardrail: Guardrail) -> None:
        self._guardrails[guardrail.id] = guardrail

    def update(self, guardrail: Guardrail) -> None:
        if guardrail.id not in self._guardrails:
            raise GuardrailNotFoundError(guardrail.id)
        self._guardrails[guardrail.id] = guardrail

    def delete(self, guardrail_id: str) -> None:
        if guardrail_id not in self._guardrails:
            raise GuardrailNotFoundError(guardrail_id)
        del self._guardrails[guardrail_id]

    def get(self, guardrail_id: str) -> Guardrail | None:
        return self._guardrails.get(guardrail_id)

    def list(self, stage=None) -> list[Guardrail]:
        values = list(self._guardrails.values())
        return [g for g in values if stage is None or g.stage == stage]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        if self.raise_on_embed is not None:
            raise self.raise_on_embed
        return [[0.0] for _ in texts]

    def search(self, vector, chunk: Chunk, stage) -> list[Match]:
        if self.raise_on_search is not None:
            raise self.raise_on_search
        return self.matches_by_text.get(chunk.evaluated_text, [])
