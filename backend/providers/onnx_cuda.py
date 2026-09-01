"""Fail-closed CUDA graph-placement validation for pinned ONNX artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from typing import Callable


CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
_MEANINGFUL_COMPUTE_OPS = frozenset(
    {
        "Attention",
        "BiasGelu",
        "EmbedLayerNormalization",
        "FusedMatMul",
        "Gelu",
        "Gemm",
        "LayerNormalization",
        "MatMul",
        "MultiHeadAttention",
        "SimplifiedLayerNormalization",
        "Softmax",
    }
)


@dataclass(frozen=True)
class NodeAssignment:
    provider: str
    name: str
    op_type: str
    domain: str


@dataclass(frozen=True)
class PlacementSummary:
    cpu_digest: str
    cpu_count: int
    cpu_operators: tuple[tuple[str, int], ...]
    cuda_operators: frozenset[str]


def enable_assignment_recording(options: object) -> None:
    add_entry = getattr(options, "add_session_config_entry", None)
    if not callable(add_entry):
        raise TypeError("ONNX SessionOptions cannot record provider assignment")
    add_entry("session.record_ep_graph_assignment_info", "1")


def _assignments(session: object) -> list[NodeAssignment]:
    get_info = getattr(session, "get_provider_graph_assignment_info", None)
    if not callable(get_info):
        raise TypeError("ONNX session cannot report provider graph assignment")
    assignments: list[NodeAssignment] = []
    for subgraph in get_info():
        provider = getattr(subgraph, "ep_name", None)
        get_nodes = getattr(subgraph, "get_nodes", None)
        if not isinstance(provider, str) or not callable(get_nodes):
            raise TypeError("ONNX provider graph assignment is malformed")
        for node in get_nodes():
            name = getattr(node, "name", None)
            op_type = getattr(node, "op_type", None)
            domain = getattr(node, "domain", None)
            if not all(isinstance(value, str) for value in (name, op_type, domain)):
                raise TypeError("ONNX provider node assignment is malformed")
            assignments.append(
                NodeAssignment(
                    provider=provider,
                    name=name,
                    op_type=op_type,
                    domain=domain,
                )
            )
    return assignments


def placement_summary(session: object) -> PlacementSummary:
    assignments = _assignments(session)
    cpu = sorted(
        (item for item in assignments if item.provider == CPU_PROVIDER),
        key=lambda item: (item.provider, item.domain, item.op_type, item.name),
    )
    serialized = [
        {
            "provider": item.provider,
            "name": item.name,
            "op_type": item.op_type,
            "domain": item.domain,
        }
        for item in cpu
    ]
    digest = hashlib.sha256(
        json.dumps(serialized, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cpu_operators = tuple(sorted(Counter(item.op_type for item in cpu).items()))
    cuda_operators = frozenset(
        item.op_type for item in assignments if item.provider == CUDA_PROVIDER
    )
    return PlacementSummary(
        cpu_digest=digest,
        cpu_count=len(cpu),
        cpu_operators=cpu_operators,
        cuda_operators=cuda_operators,
    )


def validate_cuda_placement(
    session: object,
    *,
    expected_cpu_digest: str,
    required_cuda_operators: frozenset[str],
    error_factory: Callable[[str], Exception],
    label: str,
) -> PlacementSummary:
    assignments = _assignments(session)
    meaningful_cpu = sorted(
        (
            item
            for item in assignments
            if item.provider == CPU_PROVIDER
            and item.op_type in _MEANINGFUL_COMPUTE_OPS
        ),
        key=lambda item: (item.op_type, item.name),
    )
    if meaningful_cpu:
        operators = ", ".join(
            sorted({item.op_type for item in meaningful_cpu})
        )
        raise error_factory(
            f"{label} assigned meaningful model compute to CPU: {operators}"
        )
    summary = placement_summary(session)
    if summary.cpu_digest != expected_cpu_digest:
        raise error_factory(
            f"{label} CPU bookkeeping assignment does not match the verified graph"
        )
    missing = sorted(required_cuda_operators - summary.cuda_operators)
    if missing:
        raise error_factory(
            f"{label} did not assign required compute operators to CUDA: "
            + ", ".join(missing)
        )
    return summary


__all__ = [
    "PlacementSummary",
    "enable_assignment_recording",
    "placement_summary",
    "validate_cuda_placement",
]
