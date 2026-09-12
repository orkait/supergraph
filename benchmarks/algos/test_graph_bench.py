"""Graph algorithm micro-benchmarks."""


from supergraph.algos.graph import (
    bfs_traverse,
    bidirectional_bfs,
    common_neighbors,
    dijkstra,
    find_all_paths,
)


class TestBfsTraverse:
    def test_1k_depth2(self, benchmark, graph_1k):
        result = benchmark(bfs_traverse, graph_1k, 0, 2)
        assert len(result) >= 1

    def test_10k_depth2(self, benchmark, graph_10k):
        result = benchmark(bfs_traverse, graph_10k, 0, 2)
        assert len(result) >= 1

    def test_10k_depth5(self, benchmark, graph_10k):
        result = benchmark(bfs_traverse, graph_10k, 0, 5)
        assert len(result) >= 1

    def test_100k_depth3(self, benchmark, graph_100k):
        result = benchmark(bfs_traverse, graph_100k, 0, 3)
        assert len(result) >= 1


class TestDijkstra:
    def test_1k(self, benchmark, graph_1k):
        benchmark(dijkstra, graph_1k, 0, 500)

    def test_10k(self, benchmark, graph_10k):
        benchmark(dijkstra, graph_10k, 0, 5000)

    def test_100k(self, benchmark, graph_100k):
        benchmark(dijkstra, graph_100k, 0, 50_000)


class TestBidirectionalBfs:
    def test_1k(self, benchmark, graph_1k):
        matT = graph_1k.T.tocsr()
        benchmark(bidirectional_bfs, graph_1k, matT, 0, 500, 10)

    def test_10k(self, benchmark, graph_10k):
        matT = graph_10k.T.tocsr()
        benchmark(bidirectional_bfs, graph_10k, matT, 0, 5000, 10)


class TestFindAllPaths:
    def test_1k_depth3(self, benchmark, graph_1k):
        benchmark(find_all_paths, graph_1k, 0, 500, 3)

    def test_1k_depth4(self, benchmark, graph_1k_dense):
        benchmark(find_all_paths, graph_1k_dense, 0, 500, 4)


class TestCommonNeighbors:
    def test_1k(self, benchmark, graph_1k):
        benchmark(common_neighbors, graph_1k, 0, 1)

    def test_10k(self, benchmark, graph_10k):
        benchmark(common_neighbors, graph_10k, 0, 1)
