import numpy as np

from supergraph.algos.eviction import needs_optimization, rank_evictable_slots


def test_needs_optimization_returns_expected_ops():
    result = needs_optimization(
        {
            "tombstone_ratio": 0.3,
            "string_bloat": 4.0,
            "live_nodes": 10,
            "dead_vectors": 2,
            "stale_edge_keys": 1,
            "cache_size": 300,
        }
    )
    assert result == ["COMPACT", "STRINGS", "VECTORS", "EDGES", "CACHE"]


def test_rank_evictable_slots_skips_protected_kinds_and_sorts_oldest_first():
    live_mask = np.array([True, True, True, False], dtype=bool)
    kind_ids = np.array([0, 1, 2, 0], dtype=np.int32)
    updated_at = np.array([30, 10, 20, 0], dtype=np.int64)
    updated_at_present = np.array([True, True, True, False], dtype=bool)
    kinds = ["user", "schema", "document"]

    ranked = rank_evictable_slots(
        live_mask=live_mask,
        kind_ids=kind_ids,
        kind_lookup=lambda i: kinds[i],
        updated_at=updated_at,
        updated_at_present=updated_at_present,
        protected_kinds={"schema"},
    )

    assert ranked == [2, 0]
