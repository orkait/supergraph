"""Top-K and full-sort primitives on column arrays.

Pure numpy argpartition / argsort helpers. No supergraph imports.
"""

import numpy as np

__all__ = [
    "topk_slot_order",
    "topk_from_column",
]


def topk_slot_order(
    values: np.ndarray,
    descending: bool,
    offset: int,
    limit: int | None,
    full_sort: bool = False,
) -> np.ndarray:
    """Return argpartition/argsort-derived index order with offset + limit applied.

    Args:
        values: float64 array. NaN / missing should already be pushed to
                +inf (asc) or -inf (desc) by the caller.
        descending: sort high → low when True.
        offset: drop this many leading positions after sorting.
        limit: keep at most this many positions. None = unlimited.
        full_sort: force full argsort instead of argpartition top-k.
                   Required when the caller needs to apply a post-filter
                   before slicing.

    Returns:
        int64 array of positions into `values`, offset+limit applied.
    """
    total = len(values)
    eff_offset = offset or 0
    eff_limit = limit if limit is not None else total

    if full_sort:
        sorted_idx = np.argsort(-values) if descending else np.argsort(values)
        return sorted_idx

    k = min(eff_offset + eff_limit, total)
    if 0 < k < total:
        if descending:
            # Partition largest k items to the end (no copy/negation overhead)
            part_idx = np.argpartition(values, -k)[-k:]
            # Sort only those k items descending
            sorted_idx = part_idx[np.argsort(-values[part_idx])]
        else:
            part_idx = np.argpartition(values, k)[:k]
            sorted_idx = part_idx[np.argsort(values[part_idx])]
    else:
        sorted_idx = np.argsort(-values) if descending else np.argsort(values)

    return sorted_idx[eff_offset:eff_offset + eff_limit]


def topk_from_column(
    slots: np.ndarray,
    column: np.ndarray,
    presence: np.ndarray,
    dtype_str: str,
    descending: bool,
    limit: int | None,
    offset: int | None,
    full_sort: bool = False,
) -> np.ndarray | None:
    """Sort a slot array by a column's values and slice by limit/offset.

    Args:
        slots: int array of candidate slot indices.
        column: full-size column array to index by `slots`.
        presence: full-size bool presence mask to index by `slots`.
        dtype_str: "int64" | "float64" | "int32_interned". Interned strings
                   are not orderable - returns None so caller falls back.
        descending: sort high → low.
        limit: retain up to this many positions after offset. None = all.
        offset: drop this many leading positions. None = 0.
        full_sort: force full argsort (e.g. when a post-filter is needed).

    Returns:
        Ordered subset of `slots`, or None if the column dtype is not
        orderable.
    """
    if dtype_str == "int32_interned":
        return None
    if len(slots) == 0:
        return slots

    values = column[slots].astype(np.float64)
    present = presence[slots]
    if descending:
        values[~present] = -np.inf
    else:
        values[~present] = np.inf

    order = topk_slot_order(
        values=values,
        descending=descending,
        offset=offset or 0,
        limit=limit,
        full_sort=full_sort,
    )
    return slots[order]
