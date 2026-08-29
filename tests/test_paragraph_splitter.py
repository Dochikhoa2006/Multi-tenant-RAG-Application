from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from backend.processing.paragraph_splitter import split_into_paragraphs


class SequenceEncoder:
    def __init__(self, vectors: Sequence[Sequence[float]]) -> None:
        self._vectors = np.asarray(vectors, dtype=float)
        self.inputs: list[str] = []

    def encode(self, sentences: Sequence[str], **_: object) -> np.ndarray:
        self.inputs = list(sentences)
        assert len(sentences) == len(self._vectors)
        return self._vectors


@pytest.mark.parametrize("text", ["", "   ", "\n\t\n"])
def test_empty_text_returns_no_paragraphs(text: str) -> None:
    assert split_into_paragraphs(text) == []


def test_one_sentence_is_returned_exactly_without_loading_encoder() -> None:
    text = "  One sentence with original spacing.  \n"
    assert split_into_paragraphs(text) == [text]


def test_one_semantic_topic_remains_one_lossless_paragraph() -> None:
    text = (
        "  Vectors encode meaning.   Embeddings support similarity search.\n"
        "Retrieval uses those vectors.  \n"
    )
    encoder = SequenceEncoder([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]])

    paragraphs = split_into_paragraphs(text, encoder=encoder)

    assert paragraphs == [text]
    assert encoder.inputs == [
        "Vectors encode meaning.",
        "Embeddings support similarity search.",
        "Retrieval uses those vectors.",
    ]


def test_multiple_topics_split_as_lossless_original_slices() -> None:
    text = (
        "Databases store structured records. Indexes accelerate queries.  \n\n"
        "- Bananas grow in tropical climates\n"
        "- Mangoes are tropical fruit\n"
    )
    encoder = SequenceEncoder(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]]
    )

    paragraphs = split_into_paragraphs(text, encoder=encoder)

    assert paragraphs == [
        "Databases store structured records. Indexes accelerate queries.  \n\n",
        "- Bananas grow in tropical climates\n- Mangoes are tropical fruit\n",
    ]
    assert "".join(paragraphs) == text


def test_boundary_rule_is_strictly_less_than_threshold() -> None:
    text = "First unit. Second unit."
    vectors = [[1.0, 0.0], [0.0, 1.0]]

    assert split_into_paragraphs(
        text,
        similarity_threshold=0.0,
        encoder=SequenceEncoder(vectors),
    ) == [text]
    split = split_into_paragraphs(
        text,
        similarity_threshold=0.0001,
        encoder=SequenceEncoder(vectors),
    )
    assert split == ["First unit. ", "Second unit."]
    assert "".join(split) == text


def test_markdown_and_newline_notes_are_independent_embedding_units() -> None:
    text = (
        "# Retrieval notes\n"
        "Dense vector search\n"
        "- BM25 keyword matching\n"
        "1. Cross-encoder reranking\n"
        "```python\n"
        "score = cosine(left, right)\n"
        "return score\n"
        "```\n"
        "Final reminder without punctuation\n"
    )
    encoder = SequenceEncoder([[1.0, 0.0]] * 6)

    assert split_into_paragraphs(text, encoder=encoder) == [text]
    assert encoder.inputs == [
        "# Retrieval notes",
        "Dense vector search",
        "- BM25 keyword matching",
        "1. Cross-encoder reranking",
        "```python score = cosine(left, right) return score ```",
        "Final reminder without punctuation",
    ]


def test_original_leading_trailing_and_interunit_whitespace_is_preserved() -> None:
    text = "\n  Alpha topic.\t\tRelated alpha. \r\n\r\nBeta topic.  \n"
    encoder = SequenceEncoder(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    )

    paragraphs = split_into_paragraphs(text, encoder=encoder)

    assert len(paragraphs) == 2
    assert paragraphs[0].endswith(" \r\n\r\n")
    assert "".join(paragraphs) == text


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_invalid_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_into_paragraphs(
            "First sentence. Second sentence.",
            similarity_threshold=threshold,
            encoder=SequenceEncoder([[1.0, 0.0], [1.0, 0.0]]),
        )
