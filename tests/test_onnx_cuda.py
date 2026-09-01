from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.providers.onnx_cuda import placement_summary, validate_cuda_placement


class FakeSubgraph:
    def __init__(self, provider: str, nodes: list[object]) -> None:
        self.ep_name = provider
        self._nodes = nodes

    def get_nodes(self) -> list[object]:
        return self._nodes


class FakeSession:
    def __init__(self, subgraphs: list[FakeSubgraph]) -> None:
        self._subgraphs = subgraphs

    def get_provider_graph_assignment_info(self) -> list[FakeSubgraph]:
        return self._subgraphs


def _node(name: str, op_type: str) -> object:
    return SimpleNamespace(name=name, op_type=op_type, domain="")


def test_cuda_placement_accepts_only_the_verified_cpu_bookkeeping_set() -> None:
    session = FakeSession(
        [
            FakeSubgraph(
                "CUDAExecutionProvider",
                [_node("compute", "MatMul"), _node("probabilities", "Softmax")],
            ),
            FakeSubgraph(
                "CPUExecutionProvider",
                [_node("shape-index", "Gather"), _node("axis", "Unsqueeze")],
            ),
        ]
    )
    expected = placement_summary(session)

    actual = validate_cuda_placement(
        session,
        expected_cpu_digest=expected.cpu_digest,
        required_cuda_operators=frozenset({"MatMul", "Softmax"}),
        error_factory=RuntimeError,
        label="test graph",
    )

    assert actual == expected
    assert actual.cpu_count == 2
    assert dict(actual.cpu_operators) == {"Gather": 1, "Unsqueeze": 1}


def test_cuda_placement_rejects_meaningful_cpu_compute() -> None:
    session = FakeSession(
        [
            FakeSubgraph("CUDAExecutionProvider", [_node("probabilities", "Softmax")]),
            FakeSubgraph("CPUExecutionProvider", [_node("projection", "MatMul")]),
        ]
    )
    expected = placement_summary(session)

    with pytest.raises(RuntimeError, match="meaningful model compute.*MatMul"):
        validate_cuda_placement(
            session,
            expected_cpu_digest=expected.cpu_digest,
            required_cuda_operators=frozenset({"Softmax"}),
            error_factory=RuntimeError,
            label="test graph",
        )


def test_cuda_placement_rejects_unexpected_cpu_bookkeeping() -> None:
    approved = FakeSession(
        [FakeSubgraph("CPUExecutionProvider", [_node("shape-index", "Gather")])]
    )
    changed = FakeSession(
        [
            FakeSubgraph(
                "CPUExecutionProvider",
                [_node("shape-index", "Gather"), _node("new-axis", "Unsqueeze")],
            )
        ]
    )

    with pytest.raises(RuntimeError, match="does not match the verified graph"):
        validate_cuda_placement(
            changed,
            expected_cpu_digest=placement_summary(approved).cpu_digest,
            required_cuda_operators=frozenset(),
            error_factory=RuntimeError,
            label="test graph",
        )


def test_cuda_placement_requires_expected_compute_on_cuda() -> None:
    session = FakeSession(
        [FakeSubgraph("CPUExecutionProvider", [_node("shape-index", "Gather")])]
    )
    expected = placement_summary(session)

    with pytest.raises(RuntimeError, match="required compute operators.*MatMul"):
        validate_cuda_placement(
            session,
            expected_cpu_digest=expected.cpu_digest,
            required_cuda_operators=frozenset({"MatMul"}),
            error_factory=RuntimeError,
            label="test graph",
        )
