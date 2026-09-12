"""Tests for supergraph.edges.EdgeMatrices."""

import numpy as np

from supergraph.core.edges import EdgeMatrices


# ── Helpers ─────────────────────────────────────────────────────────

def _simple_graph():
    """Build a 5-node graph with 'calls' and 'imports' edge types.

    calls:   0->1, 0->2, 1->3
    imports: 0->3, 2->4
    """
    em = EdgeMatrices()
    edges = {
        "calls": [
            (0, 1, {"line": 10}),
            (0, 2, {"line": 20}),
            (1, 3, {"line": 30}),
        ],
        "imports": [
            (0, 3, {"module": "os"}),
            (2, 4, {"module": "sys"}),
        ],
    }
    em.rebuild(edges, num_nodes=5)
    return em


# ── Empty EdgeMatrices ──────────────────────────────────────────────

class TestEmpty:
    """All methods return None or empty on a fresh instance."""

    def test_edge_types_empty(self):
        em = EdgeMatrices()
        assert em.edge_types == []

    def test_total_edges_zero(self):
        em = EdgeMatrices()
        assert em.total_edges == 0

    def test_get_none_returns_none(self):
        em = EdgeMatrices()
        assert em.get(None) is None

    def test_get_single_type_returns_none(self):
        em = EdgeMatrices()
        assert em.get({"calls"}) is None

    def test_get_multi_type_returns_none(self):
        em = EdgeMatrices()
        assert em.get({"calls", "imports"}) is None

    def test_get_transpose_returns_none(self):
        em = EdgeMatrices()
        assert em.get_transpose("calls") is None

    def test_get_edge_data_returns_empty(self):
        em = EdgeMatrices()
        assert em.get_edge_data("calls") == []

    def test_out_degree_none_returns_none(self):
        em = EdgeMatrices()
        assert em.out_degree(None) is None

    def test_out_degree_typed_returns_none(self):
        em = EdgeMatrices()
        assert em.out_degree("calls") is None

    def test_in_degree_returns_none(self):
        em = EdgeMatrices()
        assert em.in_degree("calls") is None

    def test_neighbors_out_empty(self):
        em = EdgeMatrices()
        result = em.neighbors_out(0)
        assert len(result) == 0
        assert result.dtype == np.int32

    def test_neighbors_in_empty(self):
        em = EdgeMatrices()
        result = em.neighbors_in(0, "calls")
        assert len(result) == 0
        assert result.dtype == np.int32


# ── Rebuild with single edge type ──────────────────────────────────

class TestRebuildSingle:
    """rebuild() with one edge type builds correct CSR."""

    def test_single_type_matrix_shape(self):
        em = EdgeMatrices()
        em.rebuild({"calls": [(0, 1, {}), (1, 2, {})]}, num_nodes=4)
        m = em.get({"calls"})
        assert m is not None
        assert m.shape == (4, 4)

    def test_single_type_nnz(self):
        em = EdgeMatrices()
        em.rebuild({"calls": [(0, 1, {}), (1, 2, {})]}, num_nodes=4)
        m = em.get({"calls"})
        assert m.nnz == 2

    def test_single_type_entries(self):
        em = EdgeMatrices()
        em.rebuild({"calls": [(0, 1, {}), (2, 3, {})]}, num_nodes=4)
        m = em.get({"calls"})
        assert m[0, 1] == 1
        assert m[2, 3] == 1
        assert m[0, 2] == 0
        assert m[3, 0] == 0

    def test_single_type_edge_types_property(self):
        em = EdgeMatrices()
        em.rebuild({"calls": [(0, 1, {})]}, num_nodes=2)
        assert em.edge_types == ["calls"]

    def test_empty_edge_list_skipped(self):
        em = EdgeMatrices()
        em.rebuild({"calls": [], "imports": [(0, 1, {})]}, num_nodes=2)
        assert "calls" not in em.edge_types
        assert "imports" in em.edge_types


# ── Rebuild with multiple edge types ───────────────────────────────

class TestRebuildMultiple:
    """Multiple edge types stored separately."""

    def test_both_types_present(self):
        em = _simple_graph()
        assert sorted(em.edge_types) == ["calls", "imports"]

    def test_calls_matrix_nnz(self):
        em = _simple_graph()
        assert em.get({"calls"}).nnz == 3

    def test_imports_matrix_nnz(self):
        em = _simple_graph()
        assert em.get({"imports"}).nnz == 2

    def test_total_edges(self):
        em = _simple_graph()
        assert em.total_edges == 5

    def test_types_are_independent(self):
        em = _simple_graph()
        calls = em.get({"calls"})
        imports = em.get({"imports"})
        # calls has 0->1 but imports does not
        assert calls[0, 1] == 1
        assert imports[0, 1] == 0
        # imports has 2->4 but calls does not
        assert imports[2, 4] == 1
        assert calls[2, 4] == 0


# ── get(None) returns combined matrix ──────────────────────────────

class TestGetCombined:
    """get(None) returns the precomputed union of all types."""

    def test_combined_not_none(self):
        em = _simple_graph()
        assert em.get(None) is not None

    def test_combined_has_all_edges(self):
        em = _simple_graph()
        combined = em.get(None)
        # calls: 0->1, 0->2, 1->3  imports: 0->3, 2->4
        assert combined[0, 1] >= 1
        assert combined[0, 2] >= 1
        assert combined[1, 3] >= 1
        assert combined[0, 3] >= 1
        assert combined[2, 4] >= 1

    def test_combined_nnz(self):
        em = _simple_graph()
        combined = em.get(None)
        # 5 unique (source, target) pairs: (0,1), (0,2), (1,3), (0,3), (2,4)
        assert combined.nnz == 5


# ── get({single_type}) ─────────────────────────────────────────────

class TestGetSingleType:
    """get({type}) returns the per-type matrix."""

    def test_returns_correct_matrix(self):
        em = _simple_graph()
        m = em.get({"calls"})
        assert m is not None
        assert m.nnz == 3

    def test_missing_type_returns_none(self):
        em = _simple_graph()
        assert em.get({"nonexistent"}) is None


# ── get({multiple_types}) uses cache ───────────────────────────────

class TestGetMultipleTypes:
    """get with multiple types combines and caches."""

    def test_combined_subset(self):
        em = _simple_graph()
        m = em.get({"calls", "imports"})
        assert m is not None
        assert m.nnz == 5

    def test_cache_hit(self):
        em = _simple_graph()
        m1 = em.get({"calls", "imports"})
        m2 = em.get({"calls", "imports"})
        assert m1 is m2  # same object from cache

    def test_partial_types_skips_missing(self):
        em = _simple_graph()
        m = em.get({"calls", "nonexistent"})
        assert m is not None
        assert m.nnz == 3  # only calls edges

    def test_all_missing_returns_none(self):
        em = _simple_graph()
        assert em.get({"foo", "bar"}) is None


# ── get_transpose() ────────────────────────────────────────────────

class TestGetTranspose:
    """Transposed matrix has rows/columns swapped."""

    def test_transpose_shape(self):
        em = _simple_graph()
        t = em.get_transpose("calls")
        assert t.shape == (5, 5)

    def test_transpose_entries(self):
        em = _simple_graph()
        t = em.get_transpose("calls")
        # Original calls: 0->1, 0->2, 1->3
        # Transpose: 1->0, 2->0, 3->1
        assert t[1, 0] == 1
        assert t[2, 0] == 1
        assert t[3, 1] == 1
        assert t[0, 1] == 0

    def test_transpose_cached(self):
        em = _simple_graph()
        t1 = em.get_transpose("calls")
        t2 = em.get_transpose("calls")
        assert t1 is t2

    def test_transpose_missing_type(self):
        em = _simple_graph()
        assert em.get_transpose("nonexistent") is None


# ── neighbors_out() ────────────────────────────────────────────────

class TestNeighborsOut:
    """Outgoing neighbor indices for a node."""

    def test_node_with_outgoing(self):
        em = _simple_graph()
        nbrs = em.neighbors_out(0, "calls")
        assert sorted(nbrs.tolist()) == [1, 2]

    def test_node_without_outgoing(self):
        em = _simple_graph()
        nbrs = em.neighbors_out(4, "calls")
        assert len(nbrs) == 0

    def test_all_types(self):
        em = _simple_graph()
        # node 0 has calls to 1,2 and imports to 3
        nbrs = em.neighbors_out(0)
        assert sorted(nbrs.tolist()) == [1, 2, 3]

    def test_returns_copy(self):
        em = _simple_graph()
        nbrs = em.neighbors_out(0, "calls")
        nbrs[0] = 999
        nbrs2 = em.neighbors_out(0, "calls")
        assert 999 not in nbrs2


# ── neighbors_in() ─────────────────────────────────────────────────

class TestNeighborsIn:
    """Incoming neighbor indices for a node."""

    def test_node_with_incoming(self):
        em = _simple_graph()
        # calls: 0->1, 0->2, 1->3  so node 3 gets incoming from 1
        nbrs = em.neighbors_in(3, "calls")
        assert nbrs.tolist() == [1]

    def test_node_with_multiple_incoming(self):
        em = EdgeMatrices()
        em.rebuild({"calls": [(0, 2, {}), (1, 2, {})]}, num_nodes=3)
        nbrs = em.neighbors_in(2, "calls")
        assert sorted(nbrs.tolist()) == [0, 1]

    def test_node_without_incoming(self):
        em = _simple_graph()
        nbrs = em.neighbors_in(0, "calls")
        assert len(nbrs) == 0

    def test_missing_type(self):
        em = _simple_graph()
        nbrs = em.neighbors_in(0, "nonexistent")
        assert len(nbrs) == 0


# ── out_degree() and in_degree() ───────────────────────────────────

class TestDegrees:
    """Degree arrays are precomputed and correct."""

    def test_out_degree_typed(self):
        em = _simple_graph()
        deg = em.out_degree("calls")
        assert deg is not None
        # calls: 0->1, 0->2, 1->3
        assert deg[0] == 2
        assert deg[1] == 1
        assert deg[2] == 0
        assert deg[3] == 0
        assert deg[4] == 0

    def test_out_degree_all(self):
        em = _simple_graph()
        deg = em.out_degree(None)
        assert deg is not None
        # node 0: calls 1,2 + imports 3 = 3
        assert deg[0] == 3
        # node 1: calls 3 = 1
        assert deg[1] == 1
        # node 2: imports 4 = 1
        assert deg[2] == 1
        assert deg[3] == 0
        assert deg[4] == 0

    def test_in_degree(self):
        em = _simple_graph()
        deg = em.in_degree("calls")
        assert deg is not None
        # calls: 0->1, 0->2, 1->3
        assert deg[0] == 0
        assert deg[1] == 1  # from 0
        assert deg[2] == 1  # from 0
        assert deg[3] == 1  # from 1
        assert deg[4] == 0

    def test_in_degree_imports(self):
        em = _simple_graph()
        deg = em.in_degree("imports")
        # imports: 0->3, 2->4
        assert deg[3] == 1
        assert deg[4] == 1
        assert deg[0] == 0

    def test_out_degree_missing_type(self):
        em = _simple_graph()
        assert em.out_degree("nonexistent") is None

    def test_in_degree_missing_type(self):
        em = _simple_graph()
        assert em.in_degree("nonexistent") is None

    def test_degree_array_length(self):
        em = _simple_graph()
        assert len(em.out_degree("calls")) == 5
        assert len(em.out_degree(None)) == 5
        assert len(em.in_degree("calls")) == 5


# ── Cache invalidation on rebuild ──────────────────────────────────

class TestCacheInvalidation:
    """rebuild() clears all caches."""

    def test_combination_cache_cleared(self):
        em = _simple_graph()
        m1 = em.get({"calls", "imports"})
        assert m1 is not None
        # Rebuild with different data
        em.rebuild({"calls": [(0, 1, {})]}, num_nodes=2)
        m2 = em.get({"calls", "imports"})
        # imports no longer exists, so only calls
        assert m2 is not None
        assert m2.nnz == 1

    def test_transpose_cache_cleared(self):
        em = _simple_graph()
        t1 = em.get_transpose("calls")
        em.rebuild({"calls": [(1, 0, {})]}, num_nodes=2)
        t2 = em.get_transpose("calls")
        assert t1 is not t2  # different object after rebuild
        assert t2[0, 1] == 1  # transpose of 1->0

    def test_degree_arrays_updated(self):
        em = _simple_graph()
        assert em.out_degree("calls")[0] == 2
        em.rebuild({"calls": [(0, 1, {})]}, num_nodes=3)
        assert em.out_degree("calls")[0] == 1

    def test_combined_all_updated(self):
        em = _simple_graph()
        assert em.get(None).nnz == 5
        em.rebuild({"calls": [(0, 1, {})]}, num_nodes=2)
        assert em.get(None).nnz == 1

    def test_rebuild_to_empty(self):
        em = _simple_graph()
        assert em.total_edges == 5
        em.rebuild({}, num_nodes=0)
        assert em.total_edges == 0
        assert em.edge_types == []
        assert em.get(None) is None


# ── Edge data ──────────────────────────────────────────────────────

class TestEdgeData:
    """Edge data dicts are preserved and accessible."""

    def test_data_preserved(self):
        em = _simple_graph()
        data = em.get_edge_data("calls")
        assert len(data) == 3
        assert data[0] == {"line": 10}
        assert data[1] == {"line": 20}
        assert data[2] == {"line": 30}

    def test_data_imports(self):
        em = _simple_graph()
        data = em.get_edge_data("imports")
        assert len(data) == 2
        assert data[0] == {"module": "os"}
        assert data[1] == {"module": "sys"}

    def test_data_missing_type(self):
        em = _simple_graph()
        assert em.get_edge_data("nonexistent") == []

    def test_data_cleared_on_rebuild(self):
        em = _simple_graph()
        assert len(em.get_edge_data("calls")) == 3
        em.rebuild({"imports": [(0, 1, {"new": True})]}, num_nodes=2)
        assert em.get_edge_data("calls") == []
        assert em.get_edge_data("imports") == [{"new": True}]


# ── total_edges and edge_types properties ──────────────────────────

class TestProperties:
    """Properties reflect current state."""

    def test_total_edges(self):
        em = _simple_graph()
        assert em.total_edges == 5

    def test_edge_types(self):
        em = _simple_graph()
        assert sorted(em.edge_types) == ["calls", "imports"]

    def test_total_edges_after_rebuild(self):
        em = _simple_graph()
        em.rebuild({"x": [(0, 1, {})]}, num_nodes=2)
        assert em.total_edges == 1

    def test_edge_types_after_rebuild(self):
        em = _simple_graph()
        em.rebuild({"x": [(0, 1, {})]}, num_nodes=2)
        assert em.edge_types == ["x"]
