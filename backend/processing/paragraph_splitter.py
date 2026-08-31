"""Embedding-based semantic paragraph splitting with lossless output slices."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Protocol, Sequence

import numpy as np

from backend.model_config import (
    CHUNKING,
    SEGMENTATION_EMBEDDING_DEVICE,
    SEGMENTATION_EMBEDDING_MODEL,
    SEGMENTATION_MODEL_PATH,
    TEXT_PROCESSING,
)


class SentenceEncoder(Protocol):
    """Minimal protocol implemented by SentenceTransformer and test doubles."""

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any:
        """Encode an ordered sequence of normalized semantic units."""


@dataclass(frozen=True)
class _SemanticUnit:
    """A semantic embedding input mapped to an exact source-text span."""

    start: int
    end: int
    embedding_text: str


_WHITESPACE = re.compile(TEXT_PROCESSING.whitespace_pattern)
_SENTENCE_BOUNDARY = re.compile(TEXT_PROCESSING.sentence_boundary_pattern)
_MARKDOWN_HEADING = re.compile(TEXT_PROCESSING.markdown_heading_pattern)
_MARKDOWN_LIST_ITEM = re.compile(TEXT_PROCESSING.markdown_list_item_pattern)
_MARKDOWN_FENCE = re.compile(TEXT_PROCESSING.markdown_fence_pattern)


def _normalize_for_embedding(text: str) -> str:
    return _WHITESPACE.sub(TEXT_PROCESSING.sentence_join_separator, text).strip()


def _append_unit(
    units: list[_SemanticUnit],
    text: str,
    start: int,
    end: int,
) -> None:
    embedding_text = _normalize_for_embedding(text[start:end])
    if embedding_text:
        units.append(_SemanticUnit(start=start, end=end, embedding_text=embedding_text))


def _line_content_end(line: str, line_start: int) -> int:
    return line_start + len(line.rstrip("\r\n"))


def _semantic_units(text: str) -> list[_SemanticUnit]:
    """Return Markdown-aware units while retaining offsets into original text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text or not text.strip():
        return []

    lines = text.splitlines(keepends=True)
    if not lines:
        lines = [text]

    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    units: list[_SemanticUnit] = []
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        line_start = line_starts[line_index]
        content_end = _line_content_end(line, line_start)
        line_content = text[line_start:content_end]

        if not line_content.strip():
            line_index += 1
            continue

        if _MARKDOWN_FENCE.match(line_content):
            block_end = content_end
            closing_index = line_index + 1
            while closing_index < len(lines):
                closing_start = line_starts[closing_index]
                closing_end = _line_content_end(lines[closing_index], closing_start)
                closing_content = text[closing_start:closing_end]
                block_end = closing_end
                if _MARKDOWN_FENCE.match(closing_content):
                    break
                closing_index += 1
            _append_unit(units, text, line_start, block_end)
            line_index = min(closing_index + 1, len(lines))
            continue

        if _MARKDOWN_HEADING.match(line_content) or _MARKDOWN_LIST_ITEM.match(
            line_content
        ):
            _append_unit(units, text, line_start, content_end)
            line_index += 1
            continue

        relative_start = 0
        for boundary in _SENTENCE_BOUNDARY.finditer(line_content):
            relative_end = boundary.start()
            _append_unit(
                units,
                text,
                line_start + relative_start,
                line_start + relative_end,
            )
            relative_start = boundary.end()
        _append_unit(units, text, line_start + relative_start, content_end)
        line_index += 1

    return units


def _validate_similarity_threshold(value: float) -> float:
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1 inclusive")
    return threshold


@lru_cache(maxsize=TEXT_PROCESSING.sentence_model_cache_size)
def _get_sentence_transformer() -> SentenceEncoder:
    configured_path = Path(SEGMENTATION_MODEL_PATH).expanduser()
    model_path = (
        configured_path.resolve()
        if configured_path.is_absolute()
        else (Path(__file__).resolve().parents[2] / configured_path).resolve()
    )
    if not model_path.is_dir():
        raise RuntimeError(
            "locally provisioned segmentation model directory does not exist: "
            f"{model_path}"
        )
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - misconfigured installation
        raise RuntimeError(
            "sentence-transformers is required for semantic text processing"
        ) from exc
    try:
        return SentenceTransformer(
            str(model_path),
            device=SEGMENTATION_EMBEDDING_DEVICE,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "locally provisioned segmentation model is invalid: "
            f"{SEGMENTATION_EMBEDDING_MODEL} at {model_path}"
        ) from exc


def _encode_sentences(
    sentences: Sequence[str],
    encoder: SentenceEncoder | None,
) -> np.ndarray:
    active_encoder = encoder if encoder is not None else _get_sentence_transformer()
    raw_embeddings = active_encoder.encode(
        list(sentences),
        convert_to_numpy=True,
        normalize_embeddings=TEXT_PROCESSING.normalize_embeddings,
        show_progress_bar=TEXT_PROCESSING.show_progress_bar,
    )
    embeddings = np.asarray(raw_embeddings, dtype=np.float64)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(sentences):
        raise ValueError("encoder must return one vector per input semantic unit")
    if embeddings.shape[1] == 0:
        raise ValueError("encoder vectors must contain at least one dimension")
    if not np.isfinite(embeddings).all():
        raise ValueError("encoder returned non-finite values")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("encoder returned a zero-length vector")
    return embeddings / norms


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def split_into_paragraphs(
    text: str,
    *,
    similarity_threshold: float | None = None,
    encoder: SentenceEncoder | None = None,
) -> list[str]:
    """Split text losslessly where adjacent-unit similarity is below threshold.

    Boundary semantics are exact: ``cosine_similarity(unit_i, unit_i+1) <
    threshold`` creates a paragraph boundary. Returned paragraphs are
    contiguous slices of ``text``; inter-unit whitespace belongs to the
    preceding paragraph.
    """

    units = _semantic_units(text)
    if not units:
        return []
    if len(units) == 1:
        return [text]

    threshold = _validate_similarity_threshold(
        CHUNKING.paragraph_threshold
        if similarity_threshold is None
        else similarity_threshold
    )
    embeddings = _encode_sentences(
        [unit.embedding_text for unit in units],
        encoder,
    )

    paragraphs: list[str] = []
    paragraph_start = 0
    for index in range(len(units) - 1):
        similarity = _cosine_similarity(embeddings[index], embeddings[index + 1])
        if similarity < threshold:
            boundary = units[index + 1].start
            paragraphs.append(text[paragraph_start:boundary])
            paragraph_start = boundary

    paragraphs.append(text[paragraph_start:])
    return paragraphs
