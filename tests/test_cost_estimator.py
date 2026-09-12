
import pytest
from supergraph.core.store import CoreStore
from supergraph.dsl.cost_estimator import (
    estimate_match_cost,
    estimate_traverse_cost,
    CostEstimate,
    DEFAULT_FRONTIER_THRESHOLD,
)
from supergraph.dsl.ast_nodes import (
    MatchPattern,
    PatternStep,
    PatternArrow,
    Condition,
)


@pytest.fixture
def low_degree_store():
    store = CoreStore()
    store.put_node("a", "fn", {})
    store.put_node("b", "fn", {})
    store.put_node("c", "fn", {})
    store.put_edge("a", "b", "calls")
    store.put_edge("b", "c", "calls")
    store._ensure_edges_built()
    return store


@pytest.fixture
def high_degree_store():
    store = CoreStore()
    hub_count = 500
    for i in range(hub_count):
        store.put_node(f"n{i}", "fn", {})
    for i in range(1, hub_count):
        store.put_edge("n0", f"n{i}", "calls")
    store._ensure_edges_built()
    return store


class TestEstimateTraverseCost:
    def test_low_degree_passes(self, low_degree_store):
        cost = estimate_traverse_cost(3, low_degree_store.edge_matrices, "calls")
        assert not cost.rejected
        assert cost.estimated_frontier > 0

    def test_high_degree_rejects_deep(self, high_degree_store):
        # But per-type: calls has 499 edges and 500 nodes -> avg ~1.0
        cost = estimate_traverse_cost(2, high_degree_store.edge_matrices, "calls")
        assert isinstance(cost, CostEstimate)
        assert len(cost.hops) == 2

    def test_empty_graph(self):
        store = CoreStore()
        cost = estimate_traverse_cost(5, store.edge_matrices, "calls")
        assert not cost.rejected
        assert cost.estimated_frontier == 0

    def test_no_edges_of_type(self, low_degree_store):
        cost = estimate_traverse_cost(3, low_degree_store.edge_matrices, "nonexistent")
        assert not cost.rejected
        assert cost.estimated_frontier == 0

    def test_high_degree_rejects_at_threshold(self):
        store = CoreStore()
        n = 100
        for i in range(n):
            store.put_node(f"n{i}", "fn", {})
        for i in range(n):
            for j in range(n):
                if i != j:
                    store.put_edge(f"n{i}", f"n{j}", "calls")
        store._ensure_edges_built()
        cost = estimate_traverse_cost(3, store.edge_matrices, "calls")
        assert cost.rejected
        assert cost.estimated_frontier > DEFAULT_FRONTIER_THRESHOLD
        assert "frontier exceeds" in cost.reason


class TestEstimateMatchCost:
    def test_low_degree_passes(self, low_degree_store):
        pattern = MatchPattern(
            steps=[
                PatternStep(bound_id="a"),
                PatternStep(variable="b"),
            ],
            arrows=[
                PatternArrow(expr=Condition(field="kind", op="=", value="calls")),
            ],
        )
        cost = estimate_match_cost(pattern, low_degree_store.edge_matrices)
        assert not cost.rejected
        assert cost.estimated_frontier > 0

    def test_high_degree_multi_hop_rejects(self):
        store = CoreStore()
        n = 100
        for i in range(n):
            store.put_node(f"n{i}", "fn", {})
        for i in range(n):
            for j in range(n):
                if i != j:
                    store.put_edge(f"n{i}", f"n{j}", "calls")
        store._ensure_edges_built()

        pattern = MatchPattern(
            steps=[
                PatternStep(bound_id="n0"),
                PatternStep(variable="x"),
                PatternStep(variable="y"),
                PatternStep(variable="z"),
            ],
            arrows=[
                PatternArrow(expr=Condition(field="kind", op="=", value="calls")),
                PatternArrow(expr=Condition(field="kind", op="=", value="calls")),
                PatternArrow(expr=Condition(field="kind", op="=", value="calls")),
            ],
        )
        cost = estimate_match_cost(pattern, store.edge_matrices)
        assert cost.rejected

    def test_empty_graph_returns_zero(self):
        store = CoreStore()
        pattern = MatchPattern(
            steps=[PatternStep(bound_id="a"), PatternStep(variable="b")],
            arrows=[PatternArrow(expr=Condition(field="kind", op="=", value="calls"))],
        )
        cost = estimate_match_cost(pattern, store.edge_matrices)
        assert not cost.rejected
        assert cost.estimated_frontier == 0

    def test_no_matching_edge_type(self, low_degree_store):
        pattern = MatchPattern(
            steps=[PatternStep(bound_id="a"), PatternStep(variable="b")],
            arrows=[PatternArrow(expr=Condition(field="kind", op="=", value="nonexistent"))],
        )
        cost = estimate_match_cost(pattern, low_degree_store.edge_matrices)
        assert not cost.rejected
        assert cost.estimated_frontier == 0


class TestCostEstimateToDict:
    def test_to_dict(self):
        cost = CostEstimate(
            rejected=True,
            estimated_frontier=200_000.0,
            reason="too big",
            hops=[{"depth": 1, "frontier_after": 200_000.0}],
        )
        d = cost.to_dict()
        assert d["rejected"] is True
        assert d["estimated_frontier"] == 200_000.0
        assert d["reason"] == "too big"
        assert len(d["hops"]) == 1
