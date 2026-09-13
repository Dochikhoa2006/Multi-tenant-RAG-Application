"""Local Ragas adapters and metric routing.

Ragas is intentionally imported only from this evaluation-only module. Tracking
and model-download behavior are disabled before those imports can occur.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import stat
from time import monotonic
from typing import Any, Literal, Protocol
from urllib.parse import urlparse
from uuid import UUID

from .models import EvaluationRecord


os.environ["RAGAS_DO_NOT_TRACK"] = "true"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

DEFAULT_JUDGE_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
EXPECTED_RAGAS_VERSION = "0.4.3"
EXPECTED_EMBEDDING_DIMENSION = 384
PRODUCTION_MODEL_IDS = frozenset(
    {"qwen3-4b-awq", "merged-granite-4.1-3b-query-rewrite"}
)


def _judge_activity_hooks(
    judge: Mapping[str, Any],
) -> tuple[Mapping[str, list[Any]] | None, int | None]:
    values = {
        "path": os.environ.get("RAG_EVAL_ACTIVITY_PATH"),
        "job_id": os.environ.get("RAG_EVAL_JOB_ID"),
        "request_id": os.environ.get("RAG_EVAL_REQUEST_ID"),
        "record_sha256": os.environ.get("RAG_EVAL_RECORD_SHA256"),
    }
    if all(value is None for value in values.values()):
        return None, None
    try:
        if any(not isinstance(value, str) or not value for value in values.values()):
            raise ValueError
        for name in ("job_id", "request_id"):
            if str(UUID(str(values[name]))) != values[name]:
                raise ValueError
        digest = str(values["record_sha256"])
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError
        path = Path(str(values["path"]))
        flags = os.O_WRONLY | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError
    except (OSError, ValueError) as exc:
        raise EvaluationSetupError("acceptance activity capture is invalid") from exc

    sequence = 0

    def write(event: str, request_sequence: int, success: bool | None) -> None:
        payload = {
            "schema_version": "1.0",
            "event": event,
            "evaluation_job_id": values["job_id"],
            "request_id": values["request_id"],
            "record_sha256": values["record_sha256"],
            "sequence": request_sequence,
            "judge_id": judge["model"],
            "judge_digest": judge["model_digest"],
            "monotonic": monotonic(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": success,
        }
        os.write(
            descriptor,
            (json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode(),
        )

    async def request_hook(request: Any) -> None:
        nonlocal sequence
        sequence += 1
        request.extensions["rag_evaluation_sequence"] = sequence
        write("start", sequence, None)

    async def response_hook(response: Any) -> None:
        request_sequence = response.request.extensions.get("rag_evaluation_sequence")
        if isinstance(request_sequence, int):
            write("end", request_sequence, 200 <= response.status_code < 300)

    return {"request": [request_hook], "response": [response_hook]}, descriptor


class EvaluationSetupError(RuntimeError):
    """Sanitized setup failure for the isolated evaluator."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    ollama_url: str = DEFAULT_OLLAMA_URL
    judge_model: str = DEFAULT_JUDGE_MODEL
    embedding_model_path: Path = _project_root() / "models" / "all-MiniLM-L6-v2"
    request_timeout_seconds: float = 300.0

    @classmethod
    def from_environment(
        cls,
        *,
        ollama_url: str | None = None,
        judge_model: str | None = None,
        embedding_model_path: str | Path | None = None,
    ) -> "EvaluationSettings":
        timeout_text = os.environ.get("RAG_EVAL_TIMEOUT_SECONDS", "300")
        try:
            timeout = float(timeout_text)
        except ValueError as exc:
            raise EvaluationSetupError("invalid evaluator timeout") from exc
        settings = cls(
            ollama_url=ollama_url
            or os.environ.get("RAG_EVAL_OLLAMA_URL", DEFAULT_OLLAMA_URL),
            judge_model=judge_model
            or os.environ.get("RAG_EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
            embedding_model_path=Path(
                embedding_model_path
                or os.environ.get(
                    "RAG_EVAL_EMBEDDING_MODEL_PATH",
                    str(_project_root() / "models" / "all-MiniLM-L6-v2"),
                )
            ),
            request_timeout_seconds=timeout,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        parsed = urlparse(self.ollama_url)
        if parsed.scheme != "http":
            raise EvaluationSetupError("Ollama URL must use local HTTP")
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise EvaluationSetupError("Ollama URL must resolve to loopback")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise EvaluationSetupError("Ollama URL must not contain credentials or parameters")
        if parsed.path.rstrip("/") not in {"", "/v1"}:
            raise EvaluationSetupError("Ollama URL path must be empty or /v1")
        if not isinstance(self.judge_model, str) or not self.judge_model.strip():
            raise EvaluationSetupError("judge model must be nonblank")
        normalized_model = self.judge_model.strip().casefold().split(":", 1)[0]
        if normalized_model in PRODUCTION_MODEL_IDS:
            raise EvaluationSetupError("a production inference model cannot be the judge")
        if (
            isinstance(self.request_timeout_seconds, bool)
            or not math.isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0
        ):
            raise EvaluationSetupError("evaluator timeout must be positive")

    @property
    def ollama_origin(self) -> str:
        parsed = urlparse(self.ollama_url)
        host = f"[{parsed.hostname}]" if parsed.hostname == "::1" else parsed.hostname
        port = f":{parsed.port}" if parsed.port is not None else ""
        return f"http://{host}{port}"

    @property
    def openai_base_url(self) -> str:
        return f"{self.ollama_origin}/v1"


@dataclass(frozen=True, slots=True)
class MetricPlan:
    name: str
    ragas_metric: str
    query_basis: Literal["original_query", "rewritten_query", "none"]
    requires_reference: bool = False
    opt_in: bool = False


METRIC_PLANS = (
    MetricPlan("faithfulness", "Faithfulness", "original_query"),
    MetricPlan("response_relevancy", "AnswerRelevancy", "original_query"),
    MetricPlan(
        "context_utilization",
        "ContextPrecisionWithoutReference",
        "rewritten_query",
    ),
    MetricPlan(
        "context_recall",
        "ContextRecall",
        "rewritten_query",
        requires_reference=True,
    ),
    MetricPlan(
        "context_precision_with_reference",
        "ContextPrecision",
        "rewritten_query",
        requires_reference=True,
    ),
    MetricPlan(
        "factual_correctness",
        "FactualCorrectness",
        "none",
        requires_reference=True,
    ),
    MetricPlan(
        "noise_sensitivity",
        "NoiseSensitivity",
        "original_query",
        requires_reference=True,
        opt_in=True,
    ),
)


class MetricBackend(Protocol):
    ragas_version: str
    judge_metadata: Mapping[str, Any]
    embedding_metadata: Mapping[str, Any]

    async def score(self, plan: MetricPlan, record: EvaluationRecord) -> float: ...

    async def close(self) -> None: ...


def metric_skip_code(
    plan: MetricPlan,
    record: EvaluationRecord,
    *,
    include_noise_sensitivity: bool,
) -> str | None:
    if plan.opt_in and not include_noise_sensitivity:
        return "NOT_REQUESTED"
    if plan.requires_reference and record.reference is None:
        return "REFERENCE_NOT_PROVIDED"
    return None


def _model_configuration_digest(model_path: Path) -> str:
    names = (
        "config.json",
        "config_sentence_transformers.json",
        "modules.json",
        "1_Pooling/config.json",
    )
    digest = hashlib.sha256()
    digest.update(b"rag-evaluation-embedding-config-v1\0")
    for name in names:
        path = model_path / name
        if not path.is_file() or path.is_symlink():
            raise EvaluationSetupError("local embedding model is incomplete")
        name_bytes = name.encode("utf-8")
        value = path.read_bytes()
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


async def _ollama_metadata(settings: EvaluationSettings) -> dict[str, str]:
    try:
        import httpx

        async with httpx.AsyncClient(
            base_url=settings.ollama_origin,
            timeout=settings.request_timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            version_response, tags_response = await asyncio.gather(
                client.get("/api/version"), client.get("/api/tags")
            )
        version_response.raise_for_status()
        tags_response.raise_for_status()
        version_payload = version_response.json()
        tags_payload = tags_response.json()
        version = version_payload.get("version")
        models = tags_payload.get("models")
        if not isinstance(version, str) or not version.strip() or not isinstance(models, list):
            raise EvaluationSetupError("invalid Ollama metadata")
        match = next(
            (
                item
                for item in models
                if isinstance(item, dict)
                and item.get("name") == settings.judge_model
            ),
            None,
        )
        if match is None:
            raise EvaluationSetupError("configured local judge model is not installed")
        model_digest = match.get("digest")
        if not isinstance(model_digest, str) or not model_digest.strip():
            raise EvaluationSetupError("local judge model has no version digest")
        return {
            "provider": "ollama",
            "model": settings.judge_model,
            "model_digest": model_digest,
            "ollama_version": version,
        }
    except EvaluationSetupError:
        raise
    except Exception as exc:
        raise EvaluationSetupError("local Ollama metadata is unavailable") from exc


def assert_collections_api_compatible() -> str:
    """Fail if the locked Ragas release no longer has the approved API."""

    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextPrecisionWithoutReference,
        ContextRecall,
        Faithfulness,
        FactualCorrectness,
        NoiseSensitivity,
    )

    version = importlib.metadata.version("ragas")
    if version != EXPECTED_RAGAS_VERSION:
        raise EvaluationSetupError("unexpected Ragas version")
    expected_parameters = {
        Faithfulness: {"user_input", "response", "retrieved_contexts"},
        AnswerRelevancy: {"user_input", "response"},
        ContextPrecisionWithoutReference: {
            "user_input",
            "response",
            "retrieved_contexts",
        },
        ContextRecall: {"user_input", "retrieved_contexts", "reference"},
        ContextPrecision: {"user_input", "reference", "retrieved_contexts"},
        FactualCorrectness: {"response", "reference"},
        NoiseSensitivity: {
            "user_input",
            "response",
            "reference",
            "retrieved_contexts",
        },
    }
    for metric_type, expected in expected_parameters.items():
        actual = set(inspect.signature(metric_type.ascore).parameters) - {"self"}
        if actual != expected:
            raise EvaluationSetupError("Ragas collections API is incompatible")
    return version


class RagasMetricBackend:
    """One-record Ragas metric backend using only local model resources."""

    def __init__(
        self,
        *,
        client: Any,
        metrics: Mapping[str, Any],
        embeddings: Any,
        ragas_version: str,
        judge_metadata: Mapping[str, Any],
        embedding_metadata: Mapping[str, Any],
        activity_descriptor: int | None = None,
    ) -> None:
        self._client = client
        self._metrics = dict(metrics)
        self._embeddings = embeddings
        self.ragas_version = ragas_version
        self.judge_metadata = dict(judge_metadata)
        self.embedding_metadata = dict(embedding_metadata)
        self._activity_descriptor = activity_descriptor

    @classmethod
    async def create(cls, settings: EvaluationSettings) -> "RagasMetricBackend":
        settings.validate()
        ragas_version = assert_collections_api_compatible()
        judge_metadata = await _ollama_metadata(settings)
        model_path = settings.embedding_model_path.expanduser().resolve(strict=False)
        if not model_path.is_dir() or model_path.is_symlink():
            raise EvaluationSetupError("local embedding model directory is unavailable")
        config_digest = _model_configuration_digest(model_path)

        client: Any | None = None
        activity_descriptor: int | None = None
        try:
            activity_hooks, activity_descriptor = _judge_activity_hooks(judge_metadata)
            import httpx
            from openai import AsyncOpenAI
            from ragas.embeddings import HuggingFaceEmbeddings
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                AnswerRelevancy,
                ContextPrecision,
                ContextPrecisionWithoutReference,
                ContextRecall,
                Faithfulness,
                FactualCorrectness,
                NoiseSensitivity,
            )

            client = AsyncOpenAI(
                api_key="ollama-local-only",
                base_url=settings.openai_base_url,
                max_retries=0,
                timeout=settings.request_timeout_seconds,
                http_client=httpx.AsyncClient(
                    trust_env=False,
                    event_hooks={} if activity_hooks is None else activity_hooks,
                ),
            )
            llm = llm_factory(
                settings.judge_model,
                provider="openai",
                client=client,
                temperature=0.0,
                max_tokens=4096,
            )
            embeddings = await asyncio.to_thread(
                HuggingFaceEmbeddings,
                model=str(model_path),
                device="cpu",
                normalize_embeddings=True,
                local_files_only=True,
            )
            dimension = embeddings.model_instance.get_sentence_embedding_dimension()
            if dimension != EXPECTED_EMBEDDING_DIMENSION:
                raise EvaluationSetupError("unexpected local embedding dimension")
            metrics = {
                "faithfulness": Faithfulness(llm=llm),
                "response_relevancy": AnswerRelevancy(
                    llm=llm, embeddings=embeddings
                ),
                "context_utilization": ContextPrecisionWithoutReference(llm=llm),
                "context_recall": ContextRecall(llm=llm),
                "context_precision_with_reference": ContextPrecision(llm=llm),
                "factual_correctness": FactualCorrectness(llm=llm, mode="f1"),
                "noise_sensitivity": NoiseSensitivity(llm=llm),
            }
            embedding_metadata = {
                "provider": "huggingface_local",
                "model": model_path.name,
                "configuration_sha256": config_digest,
                "dimension": dimension,
                "sentence_transformers_version": importlib.metadata.version(
                    "sentence-transformers"
                ),
            }
            return cls(
                client=client,
                metrics=metrics,
                embeddings=embeddings,
                ragas_version=ragas_version,
                judge_metadata=judge_metadata,
                embedding_metadata=embedding_metadata,
                activity_descriptor=activity_descriptor,
            )
        except EvaluationSetupError:
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass
            if activity_descriptor is not None:
                os.close(activity_descriptor)
            raise
        except Exception as exc:
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass
            if activity_descriptor is not None:
                os.close(activity_descriptor)
            raise EvaluationSetupError("local Ragas runtime initialization failed") from exc

    async def score(self, plan: MetricPlan, record: EvaluationRecord) -> float:
        metric = self._metrics[plan.name]
        contexts = list(record.retrieved_contexts)
        if plan.name == "faithfulness":
            result = await metric.ascore(
                user_input=record.original_query,
                response=record.response,
                retrieved_contexts=contexts,
            )
        elif plan.name == "response_relevancy":
            result = await metric.ascore(
                user_input=record.original_query,
                response=record.response,
            )
        elif plan.name == "context_utilization":
            result = await metric.ascore(
                user_input=record.rewritten_query,
                response=record.response,
                retrieved_contexts=contexts,
            )
        elif plan.name == "context_recall":
            result = await metric.ascore(
                user_input=record.rewritten_query,
                retrieved_contexts=contexts,
                reference=record.reference,
            )
        elif plan.name == "context_precision_with_reference":
            result = await metric.ascore(
                user_input=record.rewritten_query,
                reference=record.reference,
                retrieved_contexts=contexts,
            )
        elif plan.name == "factual_correctness":
            result = await metric.ascore(
                response=record.response,
                reference=record.reference,
            )
        elif plan.name == "noise_sensitivity":
            result = await metric.ascore(
                user_input=record.original_query,
                response=record.response,
                reference=record.reference,
                retrieved_contexts=contexts,
            )
        else:  # pragma: no cover - plans are a closed internal tuple
            raise ValueError("unsupported metric plan")
        return float(result.value)

    async def close(self) -> None:
        try:
            await self._client.close()
        finally:
            if self._activity_descriptor is not None:
                os.close(self._activity_descriptor)
                self._activity_descriptor = None

    async def smoke_embedding(self) -> int:
        vector = await asyncio.to_thread(
            self._embeddings.embed_text,
            "local evaluation compatibility smoke test",
        )
        return len(vector)


async def compatibility_smoke(settings: EvaluationSettings) -> dict[str, Any]:
    """Verify the locked API and local resources without a judge completion."""

    backend = await RagasMetricBackend.create(settings)
    try:
        dimension = await backend.smoke_embedding()
        if dimension != EXPECTED_EMBEDDING_DIMENSION:
            raise EvaluationSetupError("local embedding smoke dimension mismatch")
        return {
            "ragas_version": backend.ragas_version,
            "judge_model": backend.judge_metadata["model"],
            "judge_model_digest": backend.judge_metadata["model_digest"],
            "embedding_model": backend.embedding_metadata["model"],
            "embedding_dimension": dimension,
            "judge_completion_issued": False,
        }
    finally:
        await backend.close()


def safe_exception_type(exc: BaseException) -> str:
    name = type(exc).__name__
    return name if name.isidentifier() else "EvaluationError"
