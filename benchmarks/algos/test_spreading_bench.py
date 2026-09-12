
import numpy as np

from supergraph.algos.spreading import spreading_activation


class TestSpreadingActivation:
    def test_1k_depth2(self, benchmark, graph_1k):
        mat_t = graph_1k.T.tocsr()
        live = np.ones(1_000, dtype=bool)
        benchmark(spreading_activation, mat_t, 0, 2, 0.7, live, None, None)

    def test_10k_depth2(self, benchmark, graph_10k):
        mat_t = graph_10k.T.tocsr()
        live = np.ones(10_000, dtype=bool)
        benchmark(spreading_activation, mat_t, 0, 2, 0.7, live, None, None)

    def test_10k_depth4(self, benchmark, graph_10k):
        mat_t = graph_10k.T.tocsr()
        live = np.ones(10_000, dtype=bool)
        benchmark(spreading_activation, mat_t, 0, 4, 0.7, live, None, None)

    def test_100k_depth3(self, benchmark, graph_100k):
        mat_t = graph_100k.T.tocsr()
        live = np.ones(100_000, dtype=bool)
        benchmark(spreading_activation, mat_t, 0, 3, 0.7, live, None, None)

    def test_10k_depth3_with_modulation(self, benchmark, graph_10k):
        mat_t = graph_10k.T.tocsr()
        live = np.ones(10_000, dtype=bool)
        importance = np.random.default_rng(7).random(10_000).astype(np.float64)
        recency = np.random.default_rng(8).random(10_000).astype(np.float64)
        benchmark(
            spreading_activation, mat_t, 0, 3, 0.7, live, importance, recency,
        )
