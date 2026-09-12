
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

DEFAULT_FRONTIER_THRESHOLD = 100_000

__all__ = [
    "DEFAULT_FRONTIER_THRESHOLD",
    "CostEstimate",
    "estimate_match_cost",
    "estimate_traverse_cost",
]


@dataclass
class CostEstimate:
    rejected: bool = False
    estimated_frontier: float = 0.0
    reason: str = ""
    hops: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rejected": self.rejected,
            "estimated_frontier": self.estimated_frontier,
            "reason": self.reason,
            "hops": self.hops,
        }


def _avg_degree(matrix) -> float:
    if matrix is None:
        return 0.0
    return matrix.nnz / max(matrix.shape[0], 1)


def estimate_match_cost(
    edge_types: list[Optional[str]],
    get_matrix: Callable[[Optional[str]], Any],
    threshold: float = DEFAULT_FRONTIER_THRESHOLD,
) -> CostEstimate:
    frontier_size = 1.0
    hops: list = []
    for edge_type in edge_types:
        matrix = get_matrix(edge_type)
        if matrix is None:
            return CostEstimate(
                rejected=False,
                estimated_frontier=0.0,
                reason="no edges of this type",
                hops=hops,
            )
        frontier_size *= _avg_degree(matrix)
        hops.append({"edge_type": edge_type, "frontier_after": frontier_size})
        if frontier_size > threshold:
            return CostEstimate(
                rejected=True,
                estimated_frontier=frontier_size,
                reason=f"frontier exceeds {threshold} at hop {len(hops)}",
                hops=hops,
            )
    return CostEstimate(
        rejected=False,
        estimated_frontier=frontier_size,
        hops=hops,
    )


def estimate_traverse_cost(
    depth: int,
    matrix,
    threshold: float = DEFAULT_FRONTIER_THRESHOLD,
) -> CostEstimate:
    if matrix is None:
        return CostEstimate(rejected=False, estimated_frontier=0.0)
    avg_degree = _avg_degree(matrix)
    frontier_size = 1.0
    hops: list = []
    for d in range(depth):
        frontier_size *= avg_degree
        hops.append({"depth": d + 1, "frontier_after": frontier_size})
        if frontier_size > threshold:
            return CostEstimate(
                rejected=True,
                estimated_frontier=frontier_size,
                reason=f"frontier exceeds {threshold} at depth {d + 1}",
                hops=hops,
            )
    return CostEstimate(
        rejected=False,
        estimated_frontier=frontier_size,
        hops=hops,
    )
