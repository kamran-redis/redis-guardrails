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
