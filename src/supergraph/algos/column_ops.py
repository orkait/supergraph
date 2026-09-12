"""Column predicate evaluation primitives.

Pure numpy mask operations over column arrays. Evaluates comparison
conditions and recursive AND/OR/NOT trees given raw column data and
presence bitmasks. No supergraph imports.
"""

from typing import Any, Callable, Optional

import numpy as np

__all__ = [
    "INT64_SENTINEL",
    "STR_SENTINEL",
    "NUM_OPS",
    "eval_mask",
    "eval_mask_in",
    "eval_and",
    "eval_or",
    "eval_not",
]


INT64_SENTINEL = np.iinfo(np.int64).min
STR_SENTINEL = np.int32(-1)


NUM_OPS: dict[str, Callable] = {
    "=": np.equal,
    "!=": np.not_equal,
    ">": np.greater,
    "<": np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
}


def eval_mask(
    col: np.ndarray,
    presence: np.ndarray,
    dtype_str: str,
    op: str,
    value: Any,
    n: int,
    intern_lookup: Callable[[str], int] | None = None,
    has_string: Callable[[str], bool] | None = None,
) -> np.ndarray | None:
    """Boolean mask for a comparison predicate on a column slice.

    Returns None when the op is unsupported for the column dtype - caller
    must fall back to Python evaluation.

    For interned string columns, ``intern_lookup`` / ``has_string`` are
    required; they translate the Python-level string value to the int32
    id stored in the column.
    """
    col = col[:n]
    presence = presence[:n]

    if value is None:
        if op == "=":
            return ~presence
        if op == "!=":
            return presence.copy()
        return None

    if dtype_str == "int32_interned":
        if not isinstance(value, str):
            return None
        if has_string is None or intern_lookup is None:
            return None
        if not has_string(value):
            if op == "=":
                return np.zeros(n, dtype=bool)
            if op == "!=":
                return presence.copy()
            return None
        int_val = intern_lookup(value)
        if op == "=":
            return (col == int_val) & presence
        if op == "!=":
            return (col != int_val) & presence
        return None

    fn = NUM_OPS.get(op)
    if fn is None:
        return None
    return fn(col, value) & presence


def eval_mask_in(
    col: np.ndarray,
    presence: np.ndarray,
    dtype_str: str,
    values: list,
    n: int,
    intern_lookup: Callable[[str], int] | None = None,
    has_string: Callable[[str], bool] | None = None,
) -> np.ndarray | None:
    """Boolean mask for an IN predicate on a column slice."""
    col = col[:n]
    presence = presence[:n]

    if dtype_str == "int32_interned":
        if intern_lookup is None or has_string is None:
            return None
        int_vals = [
            intern_lookup(v)
            for v in values
            if isinstance(v, str) and has_string(v)
        ]
        if not int_vals:
            return np.zeros(n, dtype=bool)
        return np.isin(col, int_vals) & presence

    return np.isin(col, values) & presence


def eval_and(masks: list[np.ndarray]) -> np.ndarray:
    """Intersect a list of bool masks. Returns the first mask if only one."""
    if not masks:
        raise ValueError("eval_and requires at least one mask")
    result = masks[0]
    for m in masks[1:]:
        result = result & m
    return result


def eval_or(masks: list[np.ndarray]) -> np.ndarray:
    """Union a list of bool masks."""
    if not masks:
        raise ValueError("eval_or requires at least one mask")
    result = masks[0].copy()
    for m in masks[1:]:
        result = result | m
    return result


def eval_not(mask: np.ndarray, presence: np.ndarray) -> np.ndarray:
    """Negate a mask, gated by field presence so nulls don't flip to True."""
    return (~mask) & presence
