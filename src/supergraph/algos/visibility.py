"""Visibility mask primitives - tombstones, TTL, retraction, context."""

import numpy as np

__all__ = [
    "build_tombstone_mask",
    "apply_ttl_mask",
    "apply_retracted_mask",
    "full_live_mask",
]


def build_tombstone_mask(tombstones, n: int) -> np.ndarray:
    """Bool mask marking tombstoned slot indices."""
    mask = np.zeros(n, dtype=bool)
    if not tombstones:
        return mask
    tomb_arr = np.fromiter(tombstones, dtype=np.int32)
    tomb_arr = tomb_arr[tomb_arr < n]
    if tomb_arr.size:
        mask[tomb_arr] = True
    return mask


def apply_ttl_mask(
    mask: np.ndarray,
    expires_col: np.ndarray,
    presence: np.ndarray,
    now_ms: int,
) -> np.ndarray:
    """Drop slots whose __expires_at__ has already passed."""
    expired = presence & (expires_col > 0) & (expires_col < now_ms)
    return mask & ~expired


def apply_retracted_mask(
    mask: np.ndarray,
    retracted_col: np.ndarray,
    presence: np.ndarray,
) -> np.ndarray:
    """Drop slots whose __retracted__ flag is set."""
    return mask & ~(presence & (retracted_col == 1))


def full_live_mask(
    node_ids: np.ndarray,
    tombstones,
    n: int,
    now_ms: int,
    expires_col: np.ndarray | None = None,
    expires_pres: np.ndarray | None = None,
    retracted_col: np.ndarray | None = None,
    retracted_pres: np.ndarray | None = None,
) -> np.ndarray:
    """Unified visibility: alive ∧ ¬tombstoned ∧ ¬ttl_expired ∧ ¬retracted."""
    mask = node_ids[:n] >= 0
    if tombstones:
        mask = mask & ~build_tombstone_mask(tombstones, n)
    if expires_col is not None and expires_pres is not None:
        mask = apply_ttl_mask(mask, expires_col, expires_pres, now_ms)
    if retracted_col is not None and retracted_pres is not None:
        mask = apply_retracted_mask(mask, retracted_col, retracted_pres)
    return mask
