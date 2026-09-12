"""Spreading activation over a directed CSR graph."""

import numpy as np
from scipy.sparse import csr_matrix

__all__ = ["spreading_activation"]


def spreading_activation(
    matrix_t: csr_matrix | None,
    cue_slot: int | np.ndarray,
    depth: int,
    decay: float,
    live_mask: np.ndarray,
    importance: np.ndarray | None = None,
    recency: np.ndarray | None = None,
    matrix_t_delta: csr_matrix | None = None,
    cue_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Iterative spreading activation over pre-transposed adjacency.

    The transposed matrix is passed in so callers can cache it. After
    `depth` hops the activation is multiplied by the importance and
    recency modulations (if provided). The cue slot's activation is
    zeroed before return so callers don't rank it.

    Args:
        matrix_t: CSR transpose of the base adjacency matrix (frozen)
        cue_slot: starting slot index, or an array of slot indices
        depth: number of spreading hops
        decay: per-hop decay factor
        live_mask: bool mask zeroing activation on tombstoned/retracted slots
        importance: optional per-slot scalar multiplier
        recency: optional per-slot scalar multiplier
        matrix_t_delta: CSR transpose of the dynamic edge buffer (L0)
        cue_scores: optional initial activation values for cue_slots

    Returns a float64 activation array of the same length as live_mask.
    """
    n = len(live_mask)
    activation = np.zeros(n, dtype=np.float32)

    cue_is_scalar = isinstance(cue_slot, int)
    if cue_is_scalar:
        cue_valid = cue_slot >= 0 and cue_slot < n
        if cue_valid:
            activation[cue_slot] = cue_scores if cue_scores is not None else 1.0
    else:
        cue_valid_mask = (cue_slot >= 0) & (cue_slot < n)
        valid_cues = cue_slot[cue_valid_mask]
        if len(valid_cues) > 0:
            if cue_scores is not None:
                activation[valid_cues] = cue_scores[cue_valid_mask]
            else:
                activation[valid_cues] = 1.0

    live_f = live_mask.astype(np.float32)
    decay_f = np.float32(decay)

    for _ in range(depth):
        spread = None
        if matrix_t is not None:
            spread = matrix_t.dot(activation) * decay_f
        if matrix_t_delta is not None:
            if spread is None:
                spread = matrix_t_delta.dot(activation) * decay_f
            else:
                spread += matrix_t_delta.dot(activation) * decay_f

        if spread is not None:
            activation += spread
        np.multiply(activation, live_f, out=activation)

    if importance is not None:
        imp_len = len(activation)
        np.multiply(activation, importance[:imp_len].astype(np.float32), out=activation)
    if recency is not None:
        rec_len = len(activation)
        np.multiply(activation, recency[:rec_len].astype(np.float32), out=activation)

    if cue_is_scalar:
        if cue_valid:
            activation[cue_slot] = 0.0
    else:
        activation[valid_cues] = 0.0

    return activation.astype(np.float64)