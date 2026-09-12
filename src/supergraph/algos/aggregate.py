
import numpy as np

__all__ = [
    "group_assign_single",
    "group_assign_multi",
    "group_count",
    "group_sum",
    "group_avg",
    "group_min",
    "group_max",
    "group_count_distinct",
]


def group_assign_single(keys: np.ndarray) -> tuple:
    return np.unique(keys, return_inverse=True)


def group_assign_multi(key_cols: list) -> tuple:
    stacked = np.column_stack(key_cols)
    return np.unique(stacked, axis=0, return_inverse=True)


def group_count(inverse: np.ndarray, num_groups: int) -> np.ndarray:
    return np.bincount(inverse, minlength=num_groups).astype(np.float64)


def group_sum(
    values: np.ndarray,
    inverse: np.ndarray,
    num_groups: int,
) -> np.ndarray:
    sums = np.zeros(num_groups, dtype=np.float64)
    np.add.at(sums, inverse, values.astype(np.float64))
    return sums


def group_avg(
    values: np.ndarray,
    inverse: np.ndarray,
    num_groups: int,
) -> np.ndarray:
    sums = group_sum(values, inverse, num_groups)
    counts = group_count(inverse, num_groups)
    return sums / np.maximum(counts, 1)


def group_min(
    values: np.ndarray,
    inverse: np.ndarray,
    num_groups: int,
) -> np.ndarray:
    mins = np.full(num_groups, np.inf, dtype=np.float64)
    np.minimum.at(mins, inverse, values.astype(np.float64))
    return mins


def group_max(
    values: np.ndarray,
    inverse: np.ndarray,
    num_groups: int,
) -> np.ndarray:
    maxs = np.full(num_groups, -np.inf, dtype=np.float64)
    np.maximum.at(maxs, inverse, values.astype(np.float64))
    return maxs


def group_count_distinct(
    values: np.ndarray,
    inverse: np.ndarray,
    num_groups: int,
) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(num_groups, dtype=np.float64)
    _, unique_idx = np.unique(np.column_stack((inverse, values)), axis=0, return_index=True)
    return np.bincount(inverse[unique_idx], minlength=num_groups).astype(np.float64)
