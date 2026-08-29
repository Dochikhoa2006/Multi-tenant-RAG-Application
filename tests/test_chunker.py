from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from backend.model_config import CHUNKING
from backend.processing.chunker import chunk_paragraph


class SequenceEncoder:
    def __init__(self, vectors: Sequence[Sequence[float]]) -> None:
        self._vectors = np.asarray(vectors, dtype=float)

    def encode(self, sentences: Sequence[str], **_: object) -> np.ndarray:
        assert len(sentences) == len(self._vectors)
        return self._vectors


class CharacterTokenizer:
    """Deterministic tokenizer: one Unicode character equals one token."""

    def encode(self, text: str) -> list[str]:
        return list(text)


def _line(label: str, total_length: int) -> str:
    padding = max(0, total_length - len(label) - 1)
    return f"{label} {'x' * padding}"


def _token_lengths(chunks: list[str]) -> list[int]:
    return [len(chunk) for chunk in chunks]


def test_empty_input_returns_no_chunks() -> None:
    assert chunk_paragraph("\n\t ", tokenizer=CharacterTokenizer()) == []


def test_one_sentence_is_preserved_exactly_without_encoder() -> None:
    paragraph = "  One short sentence remains unchanged.  \n"

    assert chunk_paragraph(paragraph, tokenizer=CharacterTokenizer()) == [paragraph]


def test_short_paragraph_remains_one_chunk() -> None:
    paragraph = "Short first note\nShort second note\n"
    encoder = SequenceEncoder([[1.0, 0.0], [0.99, 0.01]])

    assert chunk_paragraph(
        paragraph,
        encoder=encoder,
        tokenizer=CharacterTokenizer(),
    ) == [paragraph]


def test_long_single_topic_paragraph_aims_for_target_size_losslessly() -> None:
    lines = [_line(f"Technical note {index}", 60) for index in range(8)]
    paragraph = "\n".join(lines)
    encoder = SequenceEncoder([[1.0, 0.0]] * len(lines))

    chunks = chunk_paragraph(
        paragraph,
        encoder=encoder,
        tokenizer=CharacterTokenizer(),
    )

    lengths = _token_lengths(chunks)
    assert len(chunks) > 1
    assert "".join(chunks) == paragraph
    assert all(length <= CHUNKING.max_tokens for length in lengths)
    assert all(length >= CHUNKING.min_tokens for length in lengths)
    assert all(abs(length - CHUNKING.target_tokens) <= 40 for length in lengths)


def test_semantic_topic_change_is_preferred_over_size_boundary() -> None:
    lines = [
        _line("Database records", 90),
        _line("Database indexes", 90),
        _line("Fruit orchards", 90),
        _line("Fruit harvests", 90),
    ]
    paragraph = "\n".join(lines)
    encoder = SequenceEncoder(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]]
    )
    expected_boundary = paragraph.index(lines[2])

    chunks = chunk_paragraph(
        paragraph,
        encoder=encoder,
        tokenizer=CharacterTokenizer(),
    )

    assert chunks[0] == paragraph[:expected_boundary]
    assert "".join(chunks) == paragraph


def test_long_single_unit_respects_hard_maximum_without_reformatting() -> None:
    paragraph = "prefix  " + ("technical-content " * 45) + " suffix\n"

    chunks = chunk_paragraph(paragraph, tokenizer=CharacterTokenizer())

    assert len(chunks) > 1
    assert "".join(chunks) == paragraph
    assert all(len(chunk) <= CHUNKING.max_tokens for chunk in chunks)


def test_target_size_behavior_splits_oversized_single_unit_near_target() -> None:
    paragraph = "z" * 530

    chunks = chunk_paragraph(paragraph, tokenizer=CharacterTokenizer())

    assert "".join(chunks) == paragraph
    assert [len(chunk) for chunk in chunks] == [220, 220, 90]


def test_small_compatible_chunk_merges_with_adjacent_chunk() -> None:
    first = _line("Compatible main unit", 200)
    second = _line("Tiny compatible tail", 40)
    paragraph = f"{first}\n{second}"
    encoder = SequenceEncoder([[1.0, 0.0], [0.99, 0.01]])

    chunks = chunk_paragraph(
        paragraph,
        encoder=encoder,
        tokenizer=CharacterTokenizer(),
    )

    assert chunks == [paragraph]


def test_tiny_semantic_topic_is_repaired_despite_strong_boundary() -> None:
    first = "tiny"
    second = _line("Large unrelated topic", 100)
    paragraph = f"{first}\n{second}"
    encoder = SequenceEncoder([[1.0, 0.0], [0.0, 1.0]])

    chunks = chunk_paragraph(
        paragraph,
        encoder=encoder,
        tokenizer=CharacterTokenizer(),
    )

    assert chunks == [paragraph]
    assert "".join(chunks) == paragraph
    assert all(len(chunk) <= CHUNKING.max_tokens for chunk in chunks)


def test_tiny_chunk_merges_with_most_similar_legal_neighbor() -> None:
    first = _line("First topic", 100)
    tiny = _line("Tiny topic", 40)
    following = _line("Following topic", 100)
    paragraph = f"{first}\n{tiny}\n{following}"
    encoder = SequenceEncoder(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.8660254038],
        ]
    )
    expected_first_end = paragraph.index(tiny)

    chunks = chunk_paragraph(
        paragraph,
        encoder=encoder,
        tokenizer=CharacterTokenizer(),
    )

    assert chunks == [paragraph[:expected_first_end], paragraph[expected_first_end:]]
    assert "".join(chunks) == paragraph
    assert all(len(chunk) <= CHUNKING.max_tokens for chunk in chunks)


def test_tiny_chunk_remains_when_neither_adjacent_merge_is_legal() -> None:
    first = _line("First large topic", 300)
    tiny = _line("Tiny isolated topic", 40)
    following = _line("Following large topic", 300)
    paragraph = f"{first}\n{tiny}\n{following}"
    encoder = SequenceEncoder([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    chunks = chunk_paragraph(
        paragraph,
        encoder=encoder,
        tokenizer=CharacterTokenizer(),
    )

    assert [len(chunk) for chunk in chunks] == [301, 41, 300]
    assert len(chunks[1]) < CHUNKING.min_tokens
    assert "".join(chunks) == paragraph
    assert all(len(chunk) <= CHUNKING.max_tokens for chunk in chunks)


def test_configurable_similarity_threshold_changes_boundary() -> None:
    first = _line("First topic", 100)
    second = _line("Second topic", 100)
    paragraph = f"{first}\n{second}"
    vectors = [[1.0, 0.0], [0.5, 0.8660254038]]

    split = chunk_paragraph(
        paragraph,
        similarity_threshold=0.6,
        encoder=SequenceEncoder(vectors),
        tokenizer=CharacterTokenizer(),
    )
    combined = chunk_paragraph(
        paragraph,
        similarity_threshold=0.4,
        encoder=SequenceEncoder(vectors),
        tokenizer=CharacterTokenizer(),
    )

    assert len(split) == 2
    assert combined == [paragraph]
    assert "".join(split) == paragraph


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_invalid_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        chunk_paragraph(
            "First unit\nSecond unit",
            similarity_threshold=threshold,
            encoder=SequenceEncoder([[1.0, 0.0], [1.0, 0.0]]),
            tokenizer=CharacterTokenizer(),
        )
