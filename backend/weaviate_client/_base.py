"""Shared validation and hybrid-query behavior for collection wrappers."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from weaviate.classes.query import Diversity, HybridFusion, MetadataQuery

from backend.config import get_collection_name
from backend.model_config import HYBRID_SEARCH
from backend.weaviate_client.client import WeaviateManager
from backend.weaviate_client.models import (
    SearchResult,
    UserIsolationError,
    WeaviateResponseError,
)


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _uuid_text(value: object, name: str) -> str:
    raw_value = _required_text(value, name)
    try:
        return str(UUID(raw_value))
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid UUID") from exc


def _positive_paragraph_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("paragraph_id must be an integer")
    if value <= 0:
        raise ValueError("paragraph_id must be greater than zero")
    return value


def _positive_top_k(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("top_k must be an integer")
    if value <= 0:
        raise ValueError("top_k must be greater than zero")
    return value


def _vector_values(vector: object, name: str = "vector") -> list[float]:
    if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
        raise TypeError(f"{name} must be a sequence of numbers")
    values: list[float] = []
    for item in vector:
        if isinstance(item, bool):
            raise TypeError(f"{name} must contain only numbers")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain only numbers") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} values must be finite")
        values.append(number)
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def _fusion_type() -> HybridFusion:
    normalized = re.sub(r"[^a-z]", "", HYBRID_SEARCH.fusion_method.lower())
    if normalized in {"relativescore", "relativescorefusion"}:
        return HybridFusion.RELATIVE_SCORE
    if normalized in {"ranked", "rankedfusion"}:
        return HybridFusion.RANKED
    raise ValueError(
        "HYBRID_FUSION_METHOD must be relativeScoreFusion or rankedFusion"
    )


def _result_vector(value: object) -> tuple[float, ...]:
    """Validate a native vector for snapshot/recovery operations."""

    if value is None:
        raise WeaviateResponseError("snapshot result is missing its native vector")
    selected = value
    if isinstance(value, Mapping):
        if len(value) != 1:
            raise WeaviateResponseError(
                "snapshot result contains zero or multiple native vectors"
            )
        selected = next(iter(value.values()))
    try:
        return tuple(_vector_values(selected, "snapshot vector"))
    except (TypeError, ValueError) as exc:
        raise WeaviateResponseError("snapshot result contains a malformed vector") from exc


def _result_score(value: object) -> float:
    if value is None or isinstance(value, bool):
        raise WeaviateResponseError("hybrid result is missing a numeric score")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise WeaviateResponseError("hybrid result contains a malformed score") from exc
    if not math.isfinite(score):
        raise WeaviateResponseError("hybrid result score must be finite")
    return score


def _result_uuid(value: object, name: str) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise WeaviateResponseError(f"hybrid result has an invalid {name}") from exc


class _CollectionBase:
    collection_type: str
    id_property: str
    return_properties: tuple[str, ...]

    def __init__(self, manager: WeaviateManager, user_id: str) -> None:
        if not isinstance(manager, WeaviateManager):
            raise TypeError("manager must be a WeaviateManager")
        self._manager = manager
        self.user_id = _required_text(user_id, "user_id")
        self.collection_name = get_collection_name(
            self.user_id,
            self.collection_type,
        )

    @property
    def _collection(self) -> Any:
        return self._manager.client.collections.use(self.collection_name)

    def _hybrid_search(
        self,
        query_text: str,
        query_vector: Sequence[float],
        top_k: int,
        *,
        diversity_limit: int | None = None,
        diversity_balance: float | None = None,
    ) -> list[SearchResult]:
        query = _required_text(query_text, "query_text")
        vector = _vector_values(query_vector, "query_vector")
        limit = _positive_top_k(top_k)
        query_options: dict[str, Any] = {
            "query": query,
            "vector": vector,
            "alpha": HYBRID_SEARCH.alpha,
            "query_properties": ["raw_text"],
            "fusion_type": _fusion_type(),
            "limit": limit,
            "include_vector": False,
            "return_metadata": MetadataQuery(score=True),
            "return_properties": list(self.return_properties),
        }
        if diversity_limit is not None:
            selected_limit = min(_positive_top_k(diversity_limit), limit)
            query_options["diversity_selection"] = Diversity.mmr(
                limit=selected_limit,
                balance=diversity_balance,
            )
        response = self._collection.query.hybrid(**query_options)
        results: list[SearchResult] = []
        for item in response.objects:
            raw_properties = getattr(item, "properties", None)
            if not isinstance(raw_properties, Mapping):
                raise WeaviateResponseError(
                    "hybrid result properties must be a mapping"
                )
            properties = dict(raw_properties)
            if properties.get("user_id") != self.user_id:
                raise UserIsolationError(
                    "hybrid result user_id does not match the bound collection user"
                )
            object_id = _result_uuid(getattr(item, "uuid", None), "object UUID")
            business_id = _result_uuid(
                properties.get(self.id_property),
                self.id_property,
            )
            if object_id != business_id:
                raise WeaviateResponseError(
                    f"hybrid result object UUID does not match {self.id_property}"
                )
            metadata = getattr(item, "metadata", None)
            score = getattr(metadata, "score", None) if metadata is not None else None
            results.append(
                SearchResult(
                    object_id=object_id,
                    properties=properties,
                    score=_result_score(score),
                )
            )
        return results
