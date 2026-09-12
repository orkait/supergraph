
import numpy as np

from supergraph.algos.compact import (
    apply_slot_remap_to_edges,
    build_live_mask,
    slot_remap_plan,
)


class TestBuildLiveMask:
    def test_10k_low_tomb(self, benchmark, node_ids_10k):
        tombs = set(range(0, 10_000, 100))
        benchmark(build_live_mask, node_ids_10k, tombs, 10_000)

    def test_100k_low_tomb(self, benchmark, node_ids_100k):
        tombs = set(range(0, 100_000, 100))
        benchmark(build_live_mask, node_ids_100k, tombs, 100_000)

    def test_100k_high_tomb(self, benchmark, node_ids_100k):
        tombs = set(range(0, 100_000, 2))
        benchmark(build_live_mask, node_ids_100k, tombs, 100_000)


class TestSlotRemapPlan:
    def test_10k_80pct(self, benchmark, live_mask_10k_80pct):
        benchmark(slot_remap_plan, live_mask_10k_80pct)

    def test_100k_80pct(self, benchmark, live_mask_100k_80pct):
        benchmark(slot_remap_plan, live_mask_100k_80pct)


class TestApplySlotRemap:
    def test_10k_edges(self, benchmark, edge_list_10k):
        n = 10_000
        rng = np.random.default_rng(100)
        live = rng.random(n) < 0.8
        old_to_new, _ = slot_remap_plan(live)
        benchmark(apply_slot_remap_to_edges, edge_list_10k, old_to_new, n)

    def test_100k_edges(self, benchmark, edge_list_100k):
        n = 100_000
        rng = np.random.default_rng(101)
        live = rng.random(n) < 0.8
        old_to_new, _ = slot_remap_plan(live)
        benchmark(apply_slot_remap_to_edges, edge_list_100k, old_to_new, n)
