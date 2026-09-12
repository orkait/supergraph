"""String-table garbage collection primitives.

Pure numpy helpers for computing the "what's still referenced" set and
the old_id → new_id remap array. The caller (core/optimizer.py::gc_strings)
applies the remap to store state in its imperative shell.
"""

import numpy as np

__all__ = [
    "collect_referenced_ids",
    "build_remap_plan",
    "apply_remap_to_array",
]


def collect_referenced_ids(
    node_ids: np.ndarray,
    node_kinds: np.ndarray,
    live_mask: np.ndarray,
    n: int,
    interned_columns: list,
    extra_ids=None,
) -> set:
    """Collect every interned-id value still referenced by live slots.

    Args:
        node_ids: per-slot interned id array (int32). Negative == empty.
        node_kinds: per-slot kind-id array (int32).
        live_mask: bool mask of live slots, length >= n.
        n: slot count.
        interned_columns: list of (col, presence) pairs for every
            int32_interned column. Only live slots' values are walked.
        extra_ids: optional iterable of extra ids to pin
            (e.g. edge type names interned elsewhere).

    Returns:
        set[int] - every interned id that must survive the GC.
    """
    live_ids = node_ids[:n][live_mask]
    live_kinds = node_kinds[:n][live_mask]
    referenced: set = set(live_ids[live_ids >= 0].tolist())
    referenced.update(live_kinds[live_kinds >= 0].tolist())
    for col, pres in interned_columns:
        col = col[:n]
        pres = pres[:n]
        referenced.update(col[pres].tolist())
    if extra_ids:
        referenced.update(extra_ids)
    return referenced


def build_remap_plan(
    referenced: set,
    old_table_len: int,
    lookup_fn,
) -> tuple:
    """Compute the compacted string list and old→new id lookup array.

    Args:
        referenced: set of old interned ids to keep.
        old_table_len: length of the old string table.
        lookup_fn: callable(old_id) -> string.

    Returns:
        (new_strings, remap_arr) where
            new_strings: list[str] ordered by sorted old id.
            remap_arr: int32 array of length old_table_len; remap_arr[old]
                       == new id for referenced old ids, unchanged elsewhere.
    """
    new_strings: list = []
    old_to_new: dict = {}
    for old_id in sorted(referenced):
        old_to_new[old_id] = len(new_strings)
        new_strings.append(lookup_fn(old_id))

    remap_arr = np.arange(old_table_len, dtype=np.int32)
    for old_id, new_id in old_to_new.items():
        remap_arr[old_id] = new_id

    return new_strings, remap_arr


def apply_remap_to_array(
    arr: np.ndarray,
    remap: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    """In-place fancy-index remap of int ids via a remap lookup array."""
    arr[valid_mask] = remap[arr[valid_mask]]
