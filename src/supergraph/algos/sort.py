
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
    total = len(values)
    eff_offset = offset or 0
    eff_limit = limit if limit is not None else total

    if full_sort:
        sorted_idx = np.argsort(-values) if descending else np.argsort(values)
        return sorted_idx

    k = min(eff_offset + eff_limit, total)
    if 0 < k < total:
        if descending:
            part_idx = np.argpartition(values, -k)[-k:]
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
