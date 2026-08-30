"""Thread-safe collection of the documented chat timing telemetry."""

from __future__ import annotations

from threading import Lock


TIMING_KEYS = (
    "original_query_embedding",
    "conversation_hybrid_search",
    "conversation_mmr_rerank",
    "query_rewrite",
    "rewritten_query_embedding",
    "knowledge_hybrid_search",
    "knowledge_cross_encoder_rerank",
    "policy_hybrid_search",
    "policy_cross_encoder_rerank",
    "prompt_construction",
    "ttft",
    "generation",
    "total_request",
)
TELEMETRY_SCHEMA_VERSION = "1.0"


class TelemetryCollector:
    def __init__(self) -> None:
        self._values = {key: 0.0 for key in TIMING_KEYS}
        self._lock = Lock()

    def observe(self, phase: str, elapsed_ms: float) -> None:
        if phase not in self._values:
            raise ValueError(f"unknown telemetry phase {phase!r}")
        value = float(elapsed_ms)
        if value < 0.0:
            raise ValueError("telemetry values must be non-negative")
        with self._lock:
            self._values[phase] += value

    def set(self, phase: str, elapsed_ms: float) -> None:
        if phase not in self._values:
            raise ValueError(f"unknown telemetry phase {phase!r}")
        value = float(elapsed_ms)
        if value < 0.0:
            raise ValueError("telemetry values must be non-negative")
        with self._lock:
            self._values[phase] = value

    def payload(self, request_id: str) -> dict[str, object]:
        with self._lock:
            timings = {
                key: round(max(0.0, self._values[key]), 3) for key in TIMING_KEYS
            }
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "request_id": request_id,
            "timings_ms": timings,
        }


__all__ = ["TELEMETRY_SCHEMA_VERSION", "TIMING_KEYS", "TelemetryCollector"]
