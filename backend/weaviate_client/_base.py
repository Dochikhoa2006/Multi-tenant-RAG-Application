"""Shared validation and hybrid-query behavior for collection wrappers."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from weaviate.classes.query import HybridFusion

from backend.config import get_collection_name
from backend.model_config import (
    HYBRID_SEARCH,
    LATEON_EMBEDDING_DIMENSION,
    LATE_INTERACTION_VECTOR_NAME,
    MMR_DIVERSITY_VECTOR_DIMENSION,
    MMR_DIVERSITY_VECTOR_NAME,
)
from backend.weaviate_client.client import WeaviateManager
from backend.weaviate_client.models import (
    HydratedSearchResult,
    SearchResult,
    UserIsolationError,
    WeaviateResponseError,
)


_LOGGER = logging.getLogger(__name__)


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _uuid_text(value: object, name: str) -> str:
    if isinstance(value, UUID):
        return str(value)
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


def _multi_vector_values(
    vector: object,
    name: str = "multi-vector",
) -> list[list[float]]:
    if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
        raise TypeError(f"{name} must be a sequence of vectors")
    rows = [_vector_values(row, f"{name} row") for row in vector]
    if not rows:
        raise ValueError(f"{name} must not be empty")
    dimensions = {len(row) for row in rows}
    if len(dimensions) != 1:
        raise ValueError(f"{name} rows must have consistent dimensions")
    if len(rows[0]) != LATEON_EMBEDDING_DIMENSION:
        raise ValueError(f"{name} rows must have exactly 128 dimensions")
    return rows


def _fusion_type() -> HybridFusion:
    normalized = re.sub(r"[^a-z]", "", HYBRID_SEARCH.fusion_method.lower())
    if normalized in {"relativescore", "relativescorefusion"}:
        return HybridFusion.RELATIVE_SCORE
    if normalized in {"ranked", "rankedfusion"}:
        return HybridFusion.RANKED
    raise ValueError(
        "HYBRID_FUSION_METHOD must be relativeScoreFusion or rankedFusion"
    )


def _result_vector(
    value: object,
    *,
    expected_dimension: int | None = None,
) -> tuple[float, ...]:
    """Validate the requested stored GTE MMR-diversity vector."""

    if value is None:
        raise WeaviateResponseError("result is missing its MMR-diversity vector")
    selected = value
    if isinstance(value, Mapping):
        if MMR_DIVERSITY_VECTOR_NAME not in value:
            raise WeaviateResponseError(
                "result does not contain the requested MMR-diversity vector"
            )
        selected = value[MMR_DIVERSITY_VECTOR_NAME]
    try:
        result = tuple(_vector_values(selected, "MMR-diversity vector"))
    except (TypeError, ValueError) as exc:
        raise WeaviateResponseError("result contains a malformed MMR vector") from exc
    if expected_dimension is not None and len(result) != expected_dimension:
        raise WeaviateResponseError(
            "result MMR-diversity vector must have exactly "
            f"{expected_dimension} dimensions"
        )
    return result


def _result_multi_vector(value: object) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, Mapping) or LATE_INTERACTION_VECTOR_NAME not in value:
        raise WeaviateResponseError(
            "result does not contain the requested late-interaction multi-vector"
        )
    try:
        return tuple(
            tuple(row)
            for row in _multi_vector_values(
                value[LATE_INTERACTION_VECTOR_NAME],
                "late-interaction multi-vector",
            )
        )
    except (TypeError, ValueError) as exc:
        raise WeaviateResponseError(
            "result contains a malformed late-interaction multi-vector"
        ) from exc


def _result_uuid(value: object, name: str) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise WeaviateResponseError(f"hybrid result has an invalid {name}") from exc


def _safe_object_id(value: object) -> str:
    """Return only a canonical UUID suitable for quarantine diagnostics."""

    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        return "unavailable"


def _log_quarantine(
    collection_type: str,
    object_id: object,
    reason: str,
) -> None:
    _LOGGER.warning(
        "Quarantined malformed retrieval object",
        extra={
            "collection_type": collection_type,
            "object_id": _safe_object_id(object_id),
            "reason": reason,
        },
    )


class _CollectionBase:
    collection_type: str
    id_property: str
    canonical_id_property: str
    return_properties: tuple[str, ...]
    search_return_properties: tuple[str, ...]
    hydration_return_properties: tuple[str, ...]
    search_property = "raw_text"
    segment_index_property: str | None = None
    hydrate_raw_text = False

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
        query_vector: Sequence[Sequence[float]],
        top_k: int,
    ) -> list[SearchResult]:
        query = _required_text(query_text, "query_text")
        vector = _multi_vector_values(query_vector, "query_vector")
        limit = _positive_top_k(top_k)
        query_options: dict[str, Any] = {
            "query": query,
            "vector": vector,
            "alpha": HYBRID_SEARCH.alpha,
            "query_properties": [self.search_property],
            "fusion_type": _fusion_type(),
            "limit": limit,
            "target_vector": LATE_INTERACTION_VECTOR_NAME,
            "include_vector": False,
            "return_properties": list(self.search_return_properties),
        }
        response = self._collection.query.hybrid(**query_options)
        objects = getattr(response, "objects", None)
        if not isinstance(objects, list):
            raise WeaviateResponseError("hybrid response objects are malformed")
        results: list[SearchResult] = []
        quarantined_object_ids: set[str] = set()
        quarantined_canonical_ids: set[str] = set()
        seen_object_ids: set[str] = set()
        for item in objects:
            raw_object_id = getattr(item, "uuid", None)
            raw_properties = getattr(item, "properties", None)
            if not isinstance(raw_properties, Mapping):
                _log_quarantine(
                    self.collection_type,
                    raw_object_id,
                    "properties_not_mapping",
                )
                continue
            # UUID-typed Weaviate properties deserialize as ``uuid.UUID`` in
            # the pinned v4 client.  Normalize them at this storage boundary
            # so provider-neutral retrieval and persistence models retain
            # their documented canonical-string identifiers.
            properties = {
                key: str(value) if isinstance(value, UUID) else value
                for key, value in raw_properties.items()
            }
            result_user_id = properties.get("user_id")
            if isinstance(result_user_id, str) and result_user_id != self.user_id:
                raise UserIsolationError(
                    "hybrid result user_id does not match the bound collection user"
                )
            if result_user_id != self.user_id:
                _log_quarantine(
                    self.collection_type,
                    raw_object_id,
                    "invalid_user_id",
                )
                continue
            recoverable_canonical_id: str | None = None
            if self.segment_index_property is not None:
                try:
                    recoverable_canonical_id = _result_uuid(
                        properties.get(self.canonical_id_property),
                        self.canonical_id_property,
                    )
                except WeaviateResponseError:
                    pass
            try:
                object_id = _result_uuid(raw_object_id, "object UUID")
            except WeaviateResponseError:
                if recoverable_canonical_id is not None:
                    quarantined_canonical_ids.add(recoverable_canonical_id)
                _log_quarantine(
                    self.collection_type,
                    raw_object_id,
                    "invalid_object_uuid",
                )
                continue
            try:
                business_id = _result_uuid(
                    properties.get(self.id_property),
                    self.id_property,
                )
            except WeaviateResponseError:
                if recoverable_canonical_id is not None:
                    quarantined_canonical_ids.add(recoverable_canonical_id)
                _log_quarantine(
                    self.collection_type,
                    object_id,
                    "invalid_business_uuid",
                )
                continue
            if object_id != business_id:
                if recoverable_canonical_id is not None:
                    quarantined_canonical_ids.add(recoverable_canonical_id)
                _log_quarantine(
                    self.collection_type,
                    object_id,
                    "business_uuid_mismatch",
                )
                continue
            try:
                canonical_id = _result_uuid(
                    properties.get(self.canonical_id_property),
                    self.canonical_id_property,
                )
            except WeaviateResponseError:
                _log_quarantine(
                    self.collection_type,
                    object_id,
                    "invalid_canonical_uuid",
                )
                continue
            try:
                retrieval_text = _required_text(
                    properties.get(self.search_property),
                    self.search_property,
                )
            except (TypeError, ValueError):
                if self.segment_index_property is not None:
                    quarantined_canonical_ids.add(canonical_id)
                _log_quarantine(
                    self.collection_type,
                    object_id,
                    "malformed_retrieval_text",
                )
                continue
            segment_index: int | None = None
            if self.segment_index_property is not None:
                value = properties.get(self.segment_index_property)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    quarantined_canonical_ids.add(canonical_id)
                    _log_quarantine(
                        self.collection_type,
                        object_id,
                        "invalid_segment_index",
                    )
                    continue
                segment_index = value
            if object_id in seen_object_ids:
                quarantined_object_ids.add(object_id)
                if self.segment_index_property is not None:
                    quarantined_canonical_ids.add(canonical_id)
                    quarantined_canonical_ids.update(
                        item.canonical_id
                        for item in results
                        if item.object_id == object_id
                    )
                _log_quarantine(
                    self.collection_type,
                    object_id,
                    "duplicate_object_uuid",
                )
                continue
            seen_object_ids.add(object_id)
            results.append(
                SearchResult(
                    object_id=object_id,
                    canonical_id=canonical_id,
                    retrieval_text=retrieval_text,
                    segment_index=segment_index,
                )
            )
        return [
            item
            for item in results
            if item.object_id not in quarantined_object_ids
            and item.canonical_id not in quarantined_canonical_ids
        ]

    def hydrate_mmr_head(
        self,
        candidates: Sequence[SearchResult],
    ) -> list[HydratedSearchResult]:
        """Fetch persisted MMR data once for an already bounded candidate head."""

        if isinstance(candidates, (str, bytes)) or not isinstance(
            candidates,
            Sequence,
        ):
            raise TypeError("candidates must be a sequence of SearchResult values")
        if not candidates:
            return []
        expected: dict[str, SearchResult] = {}
        for candidate in candidates:
            if not isinstance(candidate, SearchResult):
                raise TypeError("candidates must contain only SearchResult values")
            object_id = _result_uuid(candidate.object_id, "candidate object UUID")
            canonical_id = _result_uuid(
                candidate.canonical_id,
                "candidate canonical UUID",
            )
            if object_id in expected:
                raise WeaviateResponseError(
                    "MMR hydration request contains duplicate object UUIDs"
                )
            if (
                self.canonical_id_property == self.id_property
                and canonical_id != object_id
            ):
                raise WeaviateResponseError(
                    "MMR hydration candidate canonical UUID is inconsistent"
                )
            expected[object_id] = candidate

        object_ids = list(expected)
        response = self._collection.query.fetch_objects_by_ids(
            object_ids,
            limit=len(object_ids),
            include_vector=[MMR_DIVERSITY_VECTOR_NAME],
            return_properties=list(self.hydration_return_properties),
        )
        objects = getattr(response, "objects", None)
        if not isinstance(objects, list):
            raise WeaviateResponseError("MMR hydration response objects are malformed")

        hydrated: dict[str, HydratedSearchResult] = {}
        for item in objects:
            raw_object_id = getattr(item, "uuid", None)
            object_id = _result_uuid(raw_object_id, "object UUID")
            if object_id not in expected:
                raise WeaviateResponseError(
                    "MMR hydration returned an unexpected object UUID"
                )
            candidate = expected[object_id]
            raw_properties = getattr(item, "properties", None)
            if not isinstance(raw_properties, Mapping):
                _log_quarantine(
                    self.collection_type,
                    object_id,
                    "hydration_properties_not_mapping",
                )
                hydrated[object_id] = HydratedSearchResult(
                    object_id=object_id,
                    canonical_id=candidate.canonical_id,
                    diversity_vector=None,
                    segment_index=candidate.segment_index,
                    quarantine_reason="hydration_properties_not_mapping",
                )
                continue
            properties = {
                key: str(value) if isinstance(value, UUID) else value
                for key, value in raw_properties.items()
            }
            result_user_id = properties.get("user_id")
            if isinstance(result_user_id, str) and result_user_id != self.user_id:
                raise UserIsolationError(
                    "MMR hydration result user_id does not match the bound user"
                )
            if result_user_id != self.user_id:
                reason = "invalid_hydration_user_id"
                _log_quarantine(self.collection_type, object_id, reason)
                hydrated[object_id] = HydratedSearchResult(
                    object_id=object_id,
                    canonical_id=candidate.canonical_id,
                    diversity_vector=None,
                    segment_index=candidate.segment_index,
                    quarantine_reason=reason,
                )
                continue
            if object_id in hydrated:
                _log_quarantine(
                    self.collection_type,
                    object_id,
                    "duplicate_hydration_object",
                )
                hydrated[object_id] = HydratedSearchResult(
                    object_id=object_id,
                    canonical_id=candidate.canonical_id,
                    diversity_vector=None,
                    segment_index=candidate.segment_index,
                    quarantine_reason="duplicate_hydration_object",
                )
                continue
            try:
                business_id = _result_uuid(
                    properties.get(self.id_property),
                    self.id_property,
                )
            except WeaviateResponseError:
                business_id = ""
            if object_id != business_id:
                reason = "hydration_business_uuid_mismatch"
                _log_quarantine(self.collection_type, object_id, reason)
                hydrated[object_id] = HydratedSearchResult(
                    object_id=object_id,
                    canonical_id=candidate.canonical_id,
                    diversity_vector=None,
                    segment_index=candidate.segment_index,
                    quarantine_reason=reason,
                )
                continue
            try:
                canonical_id = _result_uuid(
                    properties.get(self.canonical_id_property),
                    self.canonical_id_property,
                )
            except WeaviateResponseError:
                canonical_id = ""
            if canonical_id != candidate.canonical_id:
                reason = "hydration_canonical_uuid_mismatch"
                _log_quarantine(self.collection_type, object_id, reason)
                hydrated[object_id] = HydratedSearchResult(
                    object_id=object_id,
                    canonical_id=candidate.canonical_id,
                    diversity_vector=None,
                    segment_index=candidate.segment_index,
                    quarantine_reason=reason,
                )
                continue
            segment_index: int | None = None
            if self.segment_index_property is not None:
                value = properties.get(self.segment_index_property)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    value = None
                if value != candidate.segment_index:
                    reason = "hydration_segment_index_mismatch"
                    _log_quarantine(self.collection_type, object_id, reason)
                    hydrated[object_id] = HydratedSearchResult(
                        object_id=object_id,
                        canonical_id=candidate.canonical_id,
                        diversity_vector=None,
                        segment_index=candidate.segment_index,
                        quarantine_reason=reason,
                    )
                    continue
                segment_index = value
            raw_text: str | None = None
            if self.hydrate_raw_text:
                try:
                    raw_text = _required_text(properties.get("raw_text"), "raw_text")
                except (TypeError, ValueError):
                    reason = "malformed_canonical_text"
                    _log_quarantine(self.collection_type, object_id, reason)
                    hydrated[object_id] = HydratedSearchResult(
                        object_id=object_id,
                        canonical_id=candidate.canonical_id,
                        diversity_vector=None,
                        segment_index=segment_index,
                        quarantine_reason=reason,
                    )
                    continue
            diversity_vector: tuple[float, ...] | None
            try:
                diversity_vector = _result_vector(
                    getattr(item, "vector", None),
                    expected_dimension=MMR_DIVERSITY_VECTOR_DIMENSION,
                )
            except WeaviateResponseError:
                diversity_vector = None
                _log_quarantine(
                    self.collection_type,
                    object_id,
                    "unusable_mmr_vector",
                )
            hydrated[object_id] = HydratedSearchResult(
                object_id=object_id,
                canonical_id=canonical_id,
                diversity_vector=diversity_vector,
                raw_text=raw_text,
                segment_index=segment_index,
            )

        for object_id in set(expected) - set(hydrated):
            candidate = expected[object_id]
            reason = "missing_hydration_object"
            _log_quarantine(self.collection_type, object_id, reason)
            hydrated[object_id] = HydratedSearchResult(
                object_id=object_id,
                canonical_id=candidate.canonical_id,
                diversity_vector=None,
                segment_index=candidate.segment_index,
                quarantine_reason=reason,
            )
        return [hydrated[object_id] for object_id in object_ids]
