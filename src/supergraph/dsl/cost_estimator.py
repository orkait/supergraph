
from supergraph.algos.cost import (
    DEFAULT_FRONTIER_THRESHOLD,
    CostEstimate,
    estimate_match_cost as _algo_estimate_match_cost,
    estimate_traverse_cost as _algo_estimate_traverse_cost,
)
from supergraph.dsl.ast_nodes import Condition


def _extract_edge_type(expr):
    if isinstance(expr, Condition) and expr.field == "kind" and expr.op == "=":
        return expr.value
    return None


def estimate_match_cost(pattern, edge_matrices, threshold=DEFAULT_FRONTIER_THRESHOLD):
    edge_types = [_extract_edge_type(arrow.expr) for arrow in pattern.arrows]

    def get_matrix(etype):
        return edge_matrices.get({etype} if etype else None)

    return _algo_estimate_match_cost(
        edge_types=edge_types,
        get_matrix=get_matrix,
        threshold=threshold,
    )


def estimate_traverse_cost(
    depth, edge_matrices, edge_type=None, threshold=DEFAULT_FRONTIER_THRESHOLD
):
    matrix = edge_matrices.get({edge_type} if edge_type else None)
    return _algo_estimate_traverse_cost(
        depth=depth,
        matrix=matrix,
        threshold=threshold,
    )


__all__ = [
    "CostEstimate",
    "DEFAULT_FRONTIER_THRESHOLD",
    "estimate_match_cost",
    "estimate_traverse_cost",
]
