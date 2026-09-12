from redis_guardrails.errors import IncompleteCoverageError
from redis_guardrails.models import Chunk, Scope

DEFAULT_MAX_CHARS = 800
DEFAULT_OVERLAP_CHARS = 100
DEFAULT_MAX_CHUNKS = 50


def chunk_text(
    text: str,
    *,
    source: Scope,
    prefix: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> list[Chunk]:
    budget = max_chars - len(prefix)
    if budget <= 0:
        raise IncompleteCoverageError(
            f"prefix of length {len(prefix)} leaves no room within max_chars={max_chars}"
        )

    if len(text) <= budget:
        return [
            Chunk(
                id=f"{source}-0",
                source=source,
                start_character=0,
                end_character=len(text),
                text=text,
                evaluated_text=prefix + text,
            )
        ]

    boundaries = _find_boundaries(text)
    chunks: list[Chunk] = []
    start = 0
    covered_until = 0  # max end_character reached by any chunk emitted so far
    index = 0
    while start < len(text):
        if index >= max_chunks:
            raise IncompleteCoverageError(
                f"text of length {len(text)} requires more than max_chunks={max_chunks} "
                f"chunks at max_chars={max_chars}"
            )

        end = min(start + budget, len(text))
        if end < len(text):
            # Search from max(start, covered_until), not just start: if we
            # search from `start` alone, a boundary already consumed by an
            # earlier chunk (within the overlap region) keeps winning as
            # "nearest" on every subsequent iteration, while `start` only
            # creeps forward by 1 each time (since end-overlap_chars stays
            # below the already-covered point) — producing dozens of
            # near-empty, redundant chunks that make zero forward progress
            # until max_chunks is exhausted. Anchoring the search past
            # covered_until forces each new chunk to end further into
            # genuinely new territory.
            boundary = _nearest_boundary(boundaries, max(start, covered_until), end)
            if boundary is None:
                raise IncompleteCoverageError(
                    "found a span with no paragraph, sentence, or word boundary "
                    f"within max_chars={max_chars}; cannot split safely"
                )
            end = boundary

        chunk_slice = text[start:end]
        chunks.append(
            Chunk(
                id=f"{source}-{index}",
                source=source,
                start_character=start,
                end_character=end,
                text=chunk_slice,
                evaluated_text=prefix + chunk_slice,
            )
        )
        covered_until = max(covered_until, end)

        if end >= len(text):
            break

        start = max(end - overlap_chars, start + 1)
        index += 1

    return chunks


def _find_boundaries(text: str) -> dict[str, list[int]]:
    paragraph = sorted({i + 2 for i in range(len(text) - 1) if text[i : i + 2] == "\n\n"})
    sentence = sorted({i + 1 for i, ch in enumerate(text) if ch in ".!?"})
    word = sorted({i + 1 for i, ch in enumerate(text) if ch == " "})
    return {"paragraph": paragraph, "sentence": sentence, "word": word}


def _nearest_boundary(boundaries: dict[str, list[int]], lower: int, end: int) -> int | None:
    for kind in ("paragraph", "sentence", "word"):
        candidates = [b for b in boundaries[kind] if lower < b <= end]
        if candidates:
            return max(candidates)
    return None
