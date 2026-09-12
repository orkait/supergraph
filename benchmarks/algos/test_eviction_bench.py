"""Eviction primitives micro-benchmarks."""

from supergraph.algos.eviction import needs_optimization, rank_evictable_slots


class TestNeedsOptimization:
    def test_noop(self, benchmark):
        health = {
            "tombstone_ratio": 0.0,
            "string_bloat": 1.0,
            "live_nodes": 1000,
            "dead_vectors": 0,
            "stale_edge_keys": 0,
            "cache_size": 10,
        }
        benchmark(needs_optimization, health)

    def test_all_fire(self, benchmark):
        health = {
            "tombstone_ratio": 0.5,
            "string_bloat": 10.0,
            "live_nodes": 1000,
            "dead_vectors": 50,
            "stale_edge_keys": 20,
            "cache_size": 1000,
        }
        benchmark(needs_optimization, health)


class TestRankEvictable:
    def test_10k(self, benchmark, eviction_inputs_10k):
        benchmark(rank_evictable_slots, **eviction_inputs_10k)

    def test_100k(self, benchmark, eviction_inputs_100k):
        benchmark(rank_evictable_slots, **eviction_inputs_100k)
