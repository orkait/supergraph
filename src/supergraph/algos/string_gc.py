
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
    arr[valid_mask] = remap[arr[valid_mask]]
