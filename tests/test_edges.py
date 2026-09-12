
import numpy as np

from supergraph.core.edges import EdgeMatrices


def _simple_graph():
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


class TestEmpty:

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


class TestRebuildSingle:

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


class TestRebuildMultiple:

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
        assert calls[0, 1] == 1
        assert imports[0, 1] == 0
        assert imports[2, 4] == 1
        assert calls[2, 4] == 0


class TestGetCombined:

    def test_combined_not_none(self):
        em = _simple_graph()
        assert em.get(None) is not None

    def test_combined_has_all_edges(self):
        em = _simple_graph()
        combined = em.get(None)
        assert combined[0, 1] >= 1
        assert combined[0, 2] >= 1
        assert combined[1, 3] >= 1
        assert combined[0, 3] >= 1
        assert combined[2, 4] >= 1

    def test_combined_nnz(self):
        em = _simple_graph()
        combined = em.get(None)
        assert combined.nnz == 5


class TestGetSingleType:

    def test_returns_correct_matrix(self):
        em = _simple_graph()
        m = em.get({"calls"})
        assert m is not None
        assert m.nnz == 3

    def test_missing_type_returns_none(self):
        em = _simple_graph()
        assert em.get({"nonexistent"}) is None


class TestGetMultipleTypes:

    def test_combined_subset(self):
        em = _simple_graph()
        m = em.get({"calls", "imports"})
        assert m is not None
        assert m.nnz == 5

    def test_cache_hit(self):
        em = _simple_graph()
        m1 = em.get({"calls", "imports"})
        m2 = em.get({"calls", "imports"})
        assert m1 is m2

    def test_partial_types_skips_missing(self):
        em = _simple_graph()
        m = em.get({"calls", "nonexistent"})
        assert m is not None
        assert m.nnz == 3

    def test_all_missing_returns_none(self):
        em = _simple_graph()
        assert em.get({"foo", "bar"}) is None


class TestGetTranspose:

    def test_transpose_shape(self):
        em = _simple_graph()
        t = em.get_transpose("calls")
        assert t.shape == (5, 5)

    def test_transpose_entries(self):
        em = _simple_graph()
        t = em.get_transpose("calls")
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


class TestNeighborsOut:

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
        nbrs = em.neighbors_out(0)
        assert sorted(nbrs.tolist()) == [1, 2, 3]

    def test_returns_copy(self):
        em = _simple_graph()
        nbrs = em.neighbors_out(0, "calls")
        nbrs[0] = 999
        nbrs2 = em.neighbors_out(0, "calls")
        assert 999 not in nbrs2


class TestNeighborsIn:

    def test_node_with_incoming(self):
        em = _simple_graph()
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


class TestDegrees:

    def test_out_degree_typed(self):
        em = _simple_graph()
        deg = em.out_degree("calls")
        assert deg is not None
        assert deg[0] == 2
        assert deg[1] == 1
        assert deg[2] == 0
        assert deg[3] == 0
        assert deg[4] == 0

    def test_out_degree_all(self):
        em = _simple_graph()
        deg = em.out_degree(None)
        assert deg is not None
        assert deg[0] == 3
        assert deg[1] == 1
        assert deg[2] == 1
        assert deg[3] == 0
        assert deg[4] == 0

    def test_in_degree(self):
        em = _simple_graph()
        deg = em.in_degree("calls")
        assert deg is not None
        assert deg[0] == 0
        assert deg[1] == 1
        assert deg[2] == 1
        assert deg[3] == 1
        assert deg[4] == 0

    def test_in_degree_imports(self):
        em = _simple_graph()
        deg = em.in_degree("imports")
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


class TestCacheInvalidation:

    def test_combination_cache_cleared(self):
        em = _simple_graph()
        m1 = em.get({"calls", "imports"})
        assert m1 is not None
        em.rebuild({"calls": [(0, 1, {})]}, num_nodes=2)
        m2 = em.get({"calls", "imports"})
        assert m2 is not None
        assert m2.nnz == 1

    def test_transpose_cache_cleared(self):
        em = _simple_graph()
        t1 = em.get_transpose("calls")
        em.rebuild({"calls": [(1, 0, {})]}, num_nodes=2)
        t2 = em.get_transpose("calls")
        assert t1 is not t2
        assert t2[0, 1] == 1

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


class TestEdgeData:

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


class TestProperties:

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
