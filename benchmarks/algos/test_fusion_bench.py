

from supergraph.algos.fusion import (
    normalize_bm25,
    recency_decay,
    rrf_fuse,
    weighted_remember_fusion,
)


class TestRrfFuse:
    def test_2_groups_100(self, benchmark):
        g1 = {f"id_{i}": i for i in range(100)}
        g2 = {f"id_{i}": i for i in range(100)}
        benchmark(rrf_fuse, [g1, g2])

    def test_2_groups_1k(self, benchmark):
        g1 = {f"id_{i}": i for i in range(1_000)}
        g2 = {f"id_{i}": i for i in range(1_000)}
        benchmark(rrf_fuse, [g1, g2])

    def test_3_groups_10k(self, benchmark):
        g1 = {f"id_{i}": i for i in range(10_000)}
        g2 = {f"id_{i}": i for i in range(10_000)}
        g3 = {f"id_{i}": i for i in range(10_000)}
        benchmark(rrf_fuse, [g1, g2, g3])


class TestNormalizeBm25:
    def test_10k(self, benchmark, bm25_raw_10k):
        benchmark(normalize_bm25, bm25_raw_10k)


class TestRecencyDecay:
    def test_10k(self, benchmark, updated_at_10k):
        ts, pres = updated_at_10k
        benchmark(recency_decay, ts, pres, 1_700_000_000_000, 30.0)


class TestWeightedFusion:
    def test_1k(self, benchmark, fusion_signals_1k):
        s = fusion_signals_1k
        benchmark(
            weighted_remember_fusion,
            s["vec"], s["bm25"], s["recency"], s["confidence"], s["recall"],
            [0.30, 0.20, 0.15, 0.20, 0.15],
        )

    def test_10k(self, benchmark, fusion_signals_10k):
        s = fusion_signals_10k
        benchmark(
            weighted_remember_fusion,
            s["vec"], s["bm25"], s["recency"], s["confidence"], s["recall"],
            [0.30, 0.20, 0.15, 0.20, 0.15],
        )
