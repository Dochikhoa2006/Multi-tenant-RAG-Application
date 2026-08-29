"""Semantic, token-aware paragraph chunking with lossless output slices."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol, Sequence

from backend.model_config import CHUNKING, TEXT_PROCESSING
from backend.processing.paragraph_splitter import (
    SentenceEncoder,
    _cosine_similarity,
    _encode_sentences,
    _semantic_units,
    _validate_similarity_threshold,
)


class Tokenizer(Protocol):
    """Minimal token-counting protocol implemented by tiktoken encodings."""

    def encode(self, text: str) -> Sequence[Any]:
        """Return tokens for ``text``."""


@dataclass(frozen=True)
class _Boundary:
    offset: int
    similarity: float


@dataclass(frozen=True)
class _ChunkSpan:
    start: int
    end: int


@lru_cache(maxsize=TEXT_PROCESSING.tokenizer_cache_size)
def _get_tokenizer() -> Tokenizer:
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - misconfigured installation
        raise RuntimeError("tiktoken is required for token-aware chunking") from exc
    return tiktoken.get_encoding(TEXT_PROCESSING.tokenizer_encoding)


def _token_count(text: str, tokenizer: Tokenizer) -> int:
    return len(tokenizer.encode(text))


def _prefix_end_for_limit(
    text: str,
    start: int,
    token_limit: int,
    tokenizer: Tokenizer,
) -> int:
    """Find a lossless character boundary whose prefix fits ``token_limit``."""

    low = start + 1
    high = len(text)
    best = start
    while low <= high:
        midpoint = (low + high) // 2
        if _token_count(text[start:midpoint], tokenizer) <= token_limit:
            best = midpoint
            low = midpoint + 1
        else:
            high = midpoint - 1

    while best > start and _token_count(text[start:best], tokenizer) > token_limit:
        best -= 1
    if best == start:
        raise ValueError("tokenizer cannot fit one source character within max_tokens")

    # Prefer a natural whitespace boundary without allowing an undersized
    # prefix. The whitespace stays in the preceding lossless slice.
    whitespace_end = best
    while whitespace_end > start and not text[whitespace_end - 1].isspace():
        whitespace_end -= 1
    if (
        whitespace_end > start
        and _token_count(text[start:whitespace_end], tokenizer)
        >= CHUNKING.min_tokens
    ):
        return whitespace_end
    return best


def _boundary_similarities(
    paragraph: str,
    encoder: SentenceEncoder | None,
) -> tuple[list[_Boundary], dict[int, float]]:
    units = _semantic_units(paragraph)
    if len(units) < 2:
        return [], {}

    embeddings = _encode_sentences(
        [unit.embedding_text for unit in units],
        encoder,
    )
    boundaries = [
        _Boundary(
            offset=units[index + 1].start,
            similarity=_cosine_similarity(embeddings[index], embeddings[index + 1]),
        )
        for index in range(len(units) - 1)
    ]
    return boundaries, {boundary.offset: boundary.similarity for boundary in boundaries}


def _select_next_end(
    paragraph: str,
    start: int,
    boundaries: list[_Boundary],
    threshold: float,
    tokenizer: Tokenizer,
) -> int:
    remaining_tokens = _token_count(paragraph[start:], tokenizer)
    available = [
        boundary
        for boundary in boundaries
        if boundary.offset > start
        and _token_count(paragraph[start : boundary.offset], tokenizer)
        <= CHUNKING.max_tokens
    ]

    # Semantic topic changes take precedence over target-size optimization.
    for boundary in available:
        if boundary.similarity < threshold:
            return boundary.offset

    size_candidates: list[tuple[int, int]] = []
    for boundary in available:
        chunk_tokens = _token_count(paragraph[start : boundary.offset], tokenizer)
        if chunk_tokens >= CHUNKING.min_tokens:
            size_candidates.append((boundary.offset, chunk_tokens))

    if remaining_tokens <= CHUNKING.max_tokens and (
        size_candidates
        or remaining_tokens <= CHUNKING.target_tokens + CHUNKING.min_tokens
    ):
        size_candidates.append((len(paragraph), remaining_tokens))

    if size_candidates:
        return min(
            size_candidates,
            key=lambda item: (abs(item[1] - CHUNKING.target_tokens), item[0]),
        )[0]

    # A unit with no usable internal boundary is split by source character
    # offsets. Splitting near the target leaves room for a non-tiny remainder.
    if remaining_tokens > CHUNKING.target_tokens + CHUNKING.min_tokens:
        token_limit = CHUNKING.target_tokens
    else:
        token_limit = CHUNKING.max_tokens
    return _prefix_end_for_limit(paragraph, start, token_limit, tokenizer)


def _merge_small_compatible_chunks(
    paragraph: str,
    spans: list[_ChunkSpan],
    boundary_similarities: dict[int, float],
    tokenizer: Tokenizer,
) -> list[_ChunkSpan]:
    merged = list(spans)
    index = 0
    while index < len(merged):
        current = merged[index]
        if (
            _token_count(paragraph[current.start : current.end], tokenizer)
            >= CHUNKING.min_tokens
        ):
            index += 1
            continue

        options: list[tuple[float, int, int, str]] = []
        if index > 0:
            previous = merged[index - 1]
            similarity = boundary_similarities.get(current.start, 1.0)
            combined_tokens = _token_count(
                paragraph[previous.start : current.end],
                tokenizer,
            )
            if combined_tokens <= CHUNKING.max_tokens:
                options.append(
                    (
                        similarity,
                        -abs(combined_tokens - CHUNKING.target_tokens),
                        1,
                        "previous",
                    )
                )

        if index + 1 < len(merged):
            following = merged[index + 1]
            similarity = boundary_similarities.get(current.end, 1.0)
            combined_tokens = _token_count(
                paragraph[current.start : following.end],
                tokenizer,
            )
            if combined_tokens <= CHUNKING.max_tokens:
                options.append(
                    (
                        similarity,
                        -abs(combined_tokens - CHUNKING.target_tokens),
                        0,
                        "following",
                    )
                )

        if not options:
            index += 1
            continue

        direction = max(options)[3]
        if direction == "previous":
            previous = merged[index - 1]
            merged[index - 1 : index + 1] = [
                _ChunkSpan(start=previous.start, end=current.end)
            ]
            index = max(0, index - 1)
        else:
            following = merged[index + 1]
            merged[index : index + 2] = [
                _ChunkSpan(start=current.start, end=following.end)
            ]
    return merged


def chunk_paragraph(
    paragraph: str,
    *,
    similarity_threshold: float | None = None,
    encoder: SentenceEncoder | None = None,
    tokenizer: Tokenizer | None = None,
) -> list[str]:
    """Return lossless semantic chunks bounded by configured token limits."""

    units = _semantic_units(paragraph)
    if not units:
        return []

    threshold = _validate_similarity_threshold(
        CHUNKING.chunk_threshold if similarity_threshold is None else similarity_threshold
    )
    active_tokenizer = tokenizer if tokenizer is not None else _get_tokenizer()
    boundaries, similarity_by_offset = _boundary_similarities(paragraph, encoder)

    spans: list[_ChunkSpan] = []
    chunk_start = 0
    while chunk_start < len(paragraph):
        chunk_end = _select_next_end(
            paragraph,
            chunk_start,
            boundaries,
            threshold,
            active_tokenizer,
        )
        if chunk_end <= chunk_start:
            raise RuntimeError("chunking did not advance through the source text")
        spans.append(_ChunkSpan(start=chunk_start, end=chunk_end))
        chunk_start = chunk_end

    spans = _merge_small_compatible_chunks(
        paragraph,
        spans,
        similarity_by_offset,
        active_tokenizer,
    )
    chunks = [paragraph[span.start : span.end] for span in spans]

    if any(
        _token_count(chunk, active_tokenizer) > CHUNKING.max_tokens
        for chunk in chunks
    ):
        raise RuntimeError("chunking produced output above the configured hard maximum")
    if "".join(chunks) != paragraph:
        raise RuntimeError("chunking failed to preserve the original paragraph")
    return chunks
