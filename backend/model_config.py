"""Typed, environment-backed model and retrieval configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re


def _env_string(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    value = default if raw_value is None else int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _env_probability(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    value = default if raw_value is None else float(raw_value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1 inclusive")
    return value


def _env_components(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    components = tuple(item.strip().lower() for item in raw_value.split(",") if item.strip())
    if not components:
        raise ValueError(f"{name} must contain at least one search component")
    return components


def _env_raw_string(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@dataclass(frozen=True)
class LLMConfig:
    model: str
    reasoning: str | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("LLM model must not be empty")
        if self.reasoning is not None and not self.reasoning.strip():
            raise ValueError("LLM reasoning must not be empty when provided")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("LLM max_output_tokens must be greater than zero")


@dataclass(frozen=True)
class HybridSearchConfig:
    components: tuple[str, ...]
    fusion_method: str
    alpha: float

    def __post_init__(self) -> None:
        if not self.components or any(not item for item in self.components):
            raise ValueError("Hybrid search must contain at least one component")
        if not self.fusion_method.strip():
            raise ValueError("Hybrid fusion_method must not be empty")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("Hybrid alpha must be between 0 and 1 inclusive")


@dataclass(frozen=True)
class RetrievalConfig:
    candidate_count: int
    strategy: str
    final_count: int
    mmr_lambda: float | None = None

    def __post_init__(self) -> None:
        if self.candidate_count <= 0 or self.final_count <= 0:
            raise ValueError("Retrieval counts must be greater than zero")
        if self.candidate_count < self.final_count:
            raise ValueError("Retrieval candidate_count must be at least final_count")
        if not self.strategy.strip():
            raise ValueError("Retrieval strategy must not be empty")
        if self.mmr_lambda is not None and not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("MMR lambda must be between 0 and 1 inclusive")


@dataclass(frozen=True)
class ChunkingConfig:
    paragraph_threshold: float
    chunk_threshold: float
    target_tokens: int
    min_tokens: int
    max_tokens: int

    def __post_init__(self) -> None:
        for name, value in (
            ("paragraph_threshold", self.paragraph_threshold),
            ("chunk_threshold", self.chunk_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1 inclusive")
        if min(self.min_tokens, self.target_tokens, self.max_tokens) <= 0:
            raise ValueError("Chunk token sizes must be greater than zero")
        if not self.min_tokens <= self.target_tokens <= self.max_tokens:
            raise ValueError("Chunk token sizes must satisfy min <= target <= max")


@dataclass(frozen=True)
class TokenBudgetConfig:
    knowledge_tokens: int
    policy_tokens: int
    total_context_tokens: int

    def __post_init__(self) -> None:
        if min(
            self.knowledge_tokens,
            self.policy_tokens,
            self.total_context_tokens,
        ) <= 0:
            raise ValueError("Token budgets must be greater than zero")
        if self.knowledge_tokens + self.policy_tokens > self.total_context_tokens:
            raise ValueError("Knowledge and policy budgets must fit in the total context budget")


@dataclass(frozen=True)
class TextProcessingConfig:
    tokenizer_encoding: str
    whitespace_pattern: str
    sentence_boundary_pattern: str
    markdown_heading_pattern: str
    markdown_list_item_pattern: str
    markdown_fence_pattern: str
    sentence_join_separator: str
    normalize_embeddings: bool
    show_progress_bar: bool
    sentence_model_cache_size: int
    tokenizer_cache_size: int

    def __post_init__(self) -> None:
        if not self.tokenizer_encoding.strip():
            raise ValueError("Tokenizer encoding must not be empty")
        for name, pattern in (
            ("whitespace_pattern", self.whitespace_pattern),
            ("sentence_boundary_pattern", self.sentence_boundary_pattern),
            ("markdown_heading_pattern", self.markdown_heading_pattern),
            ("markdown_list_item_pattern", self.markdown_list_item_pattern),
            ("markdown_fence_pattern", self.markdown_fence_pattern),
        ):
            if not pattern:
                raise ValueError(f"{name} must not be empty")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"{name} must be a valid regular expression") from exc
        if min(self.sentence_model_cache_size, self.tokenizer_cache_size) <= 0:
            raise ValueError("Text-processing cache sizes must be greater than zero")


PRIMARY_GENERATOR = LLMConfig(
    model=_env_string("PRIMARY_GENERATOR_MODEL", "GPT-5.1"),
    reasoning=_env_string("PRIMARY_GENERATOR_REASONING", "low"),
    max_output_tokens=_env_int("PRIMARY_GENERATOR_MAX_OUTPUT_TOKENS", 1800),
)
QUERY_REWRITER = LLMConfig(
    model=_env_string("QUERY_REWRITER_MODEL", "GPT-5 mini"),
)
SESSION_TITLE_GENERATOR = LLMConfig(
    model=_env_string("SESSION_TITLE_MODEL", "GPT-5 mini"),
)

EMBEDDING_MODEL = _env_string("EMBEDDING_MODEL", "text-embedding-3-small")
RERANKER_MODEL = _env_string("RERANKER_MODEL", "Cohere rerank-v4.0-fast")
SEGMENTATION_EMBEDDING_MODEL = _env_string(
    "SEGMENTATION_EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

HYBRID_SEARCH = HybridSearchConfig(
    components=_env_components("HYBRID_COMPONENTS", ("dense", "bm25")),
    fusion_method=_env_string("HYBRID_FUSION_METHOD", "relativeScoreFusion"),
    alpha=_env_probability("HYBRID_ALPHA", 0.70),
)

CONVERSATION_SEARCH = RetrievalConfig(
    candidate_count=_env_int("CONVERSATION_CANDIDATE_COUNT", 20),
    strategy=_env_string("CONVERSATION_RERANKING_STRATEGY", "mmr"),
    final_count=_env_int("CONVERSATION_FINAL_COUNT", 5),
    mmr_lambda=_env_probability("CONVERSATION_MMR_LAMBDA", 0.70),
)
KNOWLEDGE_SEARCH = RetrievalConfig(
    candidate_count=_env_int("KNOWLEDGE_CANDIDATE_COUNT", 30),
    strategy=_env_string("KNOWLEDGE_RERANKING_STRATEGY", "cross_encoder"),
    final_count=_env_int("KNOWLEDGE_FINAL_COUNT", 8),
)
POLICY_SEARCH = RetrievalConfig(
    candidate_count=_env_int("POLICY_CANDIDATE_COUNT", 20),
    strategy=_env_string("POLICY_RERANKING_STRATEGY", "cross_encoder"),
    final_count=_env_int("POLICY_FINAL_COUNT", 5),
)

CHUNKING = ChunkingConfig(
    paragraph_threshold=_env_probability("PARAGRAPH_SIMILARITY_THRESHOLD", 0.72),
    chunk_threshold=_env_probability("CHUNK_SIMILARITY_THRESHOLD", 0.76),
    target_tokens=_env_int("TARGET_CHUNK_TOKENS", 220),
    min_tokens=_env_int("MIN_CHUNK_TOKENS", 80),
    max_tokens=_env_int("MAX_CHUNK_TOKENS", 320),
)

TOKEN_BUDGETS = TokenBudgetConfig(
    knowledge_tokens=_env_int("KNOWLEDGE_CONTEXT_TOKENS", 4000),
    policy_tokens=_env_int("POLICY_CONTEXT_TOKENS", 1500),
    total_context_tokens=_env_int("TOTAL_CONTEXT_TOKENS", 6000),
)

TEXT_PROCESSING = TextProcessingConfig(
    tokenizer_encoding=_env_string("TOKENIZER_ENCODING", "cl100k_base"),
    whitespace_pattern=_env_string("TEXT_WHITESPACE_PATTERN", r"\s+"),
    sentence_boundary_pattern=_env_string(
        "SENTENCE_BOUNDARY_PATTERN",
        r"(?<=[.!?])\s+",
    ),
    markdown_heading_pattern=_env_string(
        "MARKDOWN_HEADING_PATTERN",
        r"^\s{0,3}#{1,6}(?:\s+|$)",
    ),
    markdown_list_item_pattern=_env_string(
        "MARKDOWN_LIST_ITEM_PATTERN",
        r"^\s*(?:[-*+]\s+|\d+[.)]\s+)",
    ),
    markdown_fence_pattern=_env_string(
        "MARKDOWN_FENCE_PATTERN",
        r"^\s*(?:```|~~~)",
    ),
    sentence_join_separator=_env_raw_string("SENTENCE_JOIN_SEPARATOR", " "),
    normalize_embeddings=_env_bool("NORMALIZE_SENTENCE_EMBEDDINGS", True),
    show_progress_bar=_env_bool("SENTENCE_EMBEDDING_PROGRESS_BAR", False),
    sentence_model_cache_size=_env_int("SENTENCE_MODEL_CACHE_SIZE", 1),
    tokenizer_cache_size=_env_int("TOKENIZER_CACHE_SIZE", 1),
)
