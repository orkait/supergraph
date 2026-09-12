import numpy as np

__all__ = ["needs_optimization", "rank_evictable_slots"]


def needs_optimization(
    health: dict,
    compact_threshold: float = 0.2,
    string_gc_threshold: float = 3.0,
    cache_gc_threshold: int = 200,
) -> list[str]:
    ops = []
    if health["tombstone_ratio"] > compact_threshold:
        ops.append("COMPACT")
    if health["string_bloat"] > string_gc_threshold and health["live_nodes"] > 0:
        ops.append("STRINGS")
    if health["dead_vectors"] > 0:
        ops.append("VECTORS")
    if health["stale_edge_keys"] > 0:
        ops.append("EDGES")
    if health["cache_size"] > cache_gc_threshold:
        ops.append("CACHE")
    return ops


def rank_evictable_slots(
    live_mask: np.ndarray,
    kind_ids: np.ndarray,
    kind_lookup,
    updated_at: np.ndarray | None,
    updated_at_present: np.ndarray | None,
    protected_kinds: set[str],
) -> list[int]:
    live_slots = np.flatnonzero(live_mask)
    if live_slots.size == 0:
        return []

    if protected_kinds:
        kind_ids_live = kind_ids[live_slots]
        unique_kinds = np.unique(kind_ids_live)

        def is_protected(k):
            if k < 0:
                return False
            try:
                return kind_lookup(int(k)) in protected_kinds
            except KeyError:
                return False

        protected_mask = np.fromiter(map(is_protected, unique_kinds), dtype=bool, count=unique_kinds.size)
        protected_ids = unique_kinds[protected_mask]

        n_protected = protected_ids.size
        if n_protected == 1:
            live_slots = live_slots[kind_ids_live != protected_ids[0]]
        elif n_protected == 2:
            live_slots = live_slots[(kind_ids_live != protected_ids[0]) & (kind_ids_live != protected_ids[1])]
        elif n_protected > 2:
            live_slots = live_slots[~np.in1d(kind_ids_live, protected_ids)]

    if live_slots.size == 0:
        return []

    if updated_at is not None and updated_at_present is not None:
        # Rank slots by "oldest touch first". Slots without __updated_at__
        # set (absent presence bit) previously got a product of 0 and sorted
        # as the oldest — meaning "never updated" was treated as "oldest",
        # which evicted fresh un-tagged nodes first (bug #98). We now treat
        # a missing timestamp as +inf so those slots sort last, matching the
        # intuitive "if we don't know when it was last touched, assume it's
        # still relevant" default.
        live_ts = updated_at[live_slots].astype(np.float64, copy=True)
        live_pres = updated_at_present[live_slots].astype(bool)
        live_ts[~live_pres] = np.inf
        sorted_indices = np.argsort(live_ts, kind="stable")
        return live_slots[sorted_indices].tolist()

    return live_slots.tolist()