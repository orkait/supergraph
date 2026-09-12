import numpy as np
from scipy.sparse import csr_matrix

from supergraph.core.path import (
    bidirectional_bfs,
    bfs_traverse,
    common_neighbors,
    find_all_paths,
)


def make_csr(edges, n):
    if not edges:
        return csr_matrix((n, n), dtype=np.int8)
    sources, targets = zip(*edges)
    return csr_matrix(
        (np.ones(len(sources), dtype=np.int8),
         (np.array(sources), np.array(targets))),
        shape=(n, n),
    )


class TestBidirectionalBfs:
    def test_direct_edge(self):
        m = make_csr([(0, 1)], 2)
        mt = m.T.tocsr()
        assert bidirectional_bfs(m, mt, 0, 1) == [0, 1]

    def test_depth_2(self):
        m = make_csr([(0, 1), (1, 2)], 3)
        mt = m.T.tocsr()
        assert bidirectional_bfs(m, mt, 0, 2) == [0, 1, 2]

    def test_depth_3(self):
        m = make_csr([(0, 1), (1, 2), (2, 3)], 4)
        mt = m.T.tocsr()
        assert bidirectional_bfs(m, mt, 0, 3) == [0, 1, 2, 3]

    def test_no_path(self):
        m = make_csr([(0, 1), (2, 3)], 4)
        mt = m.T.tocsr()
        assert bidirectional_bfs(m, mt, 0, 3) is None

    def test_same_source_target(self):
        m = make_csr([(0, 1)], 2)
        mt = m.T.tocsr()
        assert bidirectional_bfs(m, mt, 0, 0) == [0]

    def test_max_depth_too_small(self):
        m = make_csr([(0, 1), (1, 2), (2, 3)], 4)
        mt = m.T.tocsr()
        assert bidirectional_bfs(m, mt, 0, 3, max_depth=1) is None

    def test_multiple_shortest_paths(self):
        m = make_csr([(0, 1), (0, 2), (1, 3), (2, 3)], 4)
        mt = m.T.tocsr()
        result = bidirectional_bfs(m, mt, 0, 3)
        assert result is not None
        assert len(result) == 3
        assert result[0] == 0
        assert result[-1] == 3

    def test_disconnected_components(self):
        m = make_csr([(0, 1), (2, 3)], 4)
        mt = m.T.tocsr()
        assert bidirectional_bfs(m, mt, 0, 2) is None

    def test_single_node_graph(self):
        m = make_csr([], 1)
        mt = m.T.tocsr()
        assert bidirectional_bfs(m, mt, 0, 0) == [0]


class TestBfsTraverse:
    def test_no_outgoing_edges(self):
        m = make_csr([(0, 1)], 2)
        assert bfs_traverse(m, 1, 5) == [(1, 0)]

    def test_depth_1(self):
        m = make_csr([(0, 1), (0, 2)], 3)
        result = bfs_traverse(m, 0, 1)
        assert (0, 0) in result
        depths = {node: d for node, d in result}
        assert depths[0] == 0
        assert depths[1] == 1
        assert depths[2] == 1

    def test_depth_2(self):
        m = make_csr([(0, 1), (1, 2)], 3)
        result = bfs_traverse(m, 0, 2)
        depths = {node: d for node, d in result}
        assert depths[0] == 0
        assert depths[1] == 1
        assert depths[2] == 2

    def test_cycle_no_infinite_loop(self):
        m = make_csr([(0, 1), (1, 2), (2, 0)], 3)
        result = bfs_traverse(m, 0, 10)
        nodes = [node for node, _ in result]
        assert sorted(nodes) == [0, 1, 2]

    def test_max_depth_0(self):
        m = make_csr([(0, 1), (0, 2)], 3)
        assert bfs_traverse(m, 0, 0) == [(0, 0)]

    def test_repeated_calls_return_same_result(self):
        m = make_csr([(0, 1), (1, 2)], 3)
        first = bfs_traverse(m, 0, 2)
        second = bfs_traverse(m, 0, 2)
        assert first == second


class TestFindAllPaths:
    def test_single_path(self):
        m = make_csr([(0, 1), (1, 2)], 3)
        paths = find_all_paths(m, 0, 2, max_depth=5)
        assert paths == [[0, 1, 2]]

    def test_multiple_paths(self):
        m = make_csr([(0, 1), (0, 2), (1, 3), (2, 3)], 4)
        paths = find_all_paths(m, 0, 3, max_depth=5)
        assert len(paths) == 2
        assert [0, 1, 3] in paths
        assert [0, 2, 3] in paths

    def test_no_path(self):
        m = make_csr([(0, 1), (2, 3)], 4)
        paths = find_all_paths(m, 0, 3, max_depth=5)
        assert paths == []

    def test_same_source_target(self):
        m = make_csr([(0, 1)], 2)
        paths = find_all_paths(m, 0, 0, max_depth=5)
        assert paths == [[0]]

    def test_max_results_caps_output(self):
        m = make_csr([(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)], 5)
        paths = find_all_paths(m, 0, 4, max_depth=5, max_results=2)
        assert len(paths) == 2

    def test_cycle_no_infinite_loop(self):
        m = make_csr([(0, 1), (1, 2), (2, 0), (2, 3)], 4)
        paths = find_all_paths(m, 0, 3, max_depth=10)
        assert len(paths) >= 1
        assert [0, 1, 2, 3] in paths


class TestCommonNeighbors:
    def test_shared_neighbors(self):
        m = make_csr([(0, 2), (0, 3), (1, 2), (1, 4)], 5)
        result = common_neighbors(m, 0, 1)
        np.testing.assert_array_equal(result, [2])

    def test_no_shared_neighbors(self):
        m = make_csr([(0, 1), (2, 3)], 4)
        result = common_neighbors(m, 0, 2)
        assert len(result) == 0

    def test_one_node_no_neighbors(self):
        m = make_csr([(0, 2)], 3)
        result = common_neighbors(m, 0, 1)
        assert len(result) == 0
