"""Score fusion primitives for hybrid retrieval."""

import numpy as np

__all__ = [
    "rrf_fuse",
    "rrf_remember_fusion",
    "normalize_bm25",
    "recency_decay",
    "temporal_proximity",
    "weighted_remember_fusion",
]


def rrf_fuse(
    rank_groups: list[dict[str, int]],
    k_rrf: float = 60.0,
) -> dict[str, float]:
    """Reciprocal Rank Fusion over N ranked lists keyed by id.

    Each entry in rank_groups is {item_id: rank_from_0}. Returns fused
    {item_id: score} summing 1 / (k_rrf + rank) across groups.
    """
    fused: dict[str, float] = {}
    fused_get = fused.get
    for group in rank_groups:
        for item_id, rank in group.items():
            fused[item_id] = fused_get(item_id, 0.0) + 1.0 / (k_rrf + rank)
    return fused


def rrf_remember_fusion(
    *signals: np.ndarray,
    candidate_slots: np.ndarray,
    k_rrf: float = 60.0,
) -> np.ndarray:
    """Reciprocal Rank Fusion over N signal arrays aligned by slot index.

    For each signal, ranks the candidate_slots by that signal's values (desc),
    then sums 1 / (k_rrf + rank) across all signals. Returns a full-length
    score array (same shape as each signal) with RRF scores at candidate slots.

    Signals with all-zero values for a candidate contribute rank = len(candidates)
    (worst rank) for that candidate, so inactive signals don't distort rankings.
    """
    n = signals[0].shape[0]
    fused = np.zeros(n, dtype=np.float64)
    n_cand = len(candidate_slots)

    if n_cand == 0:
        return fused

    k_rrf = max(k_rrf, 1.0)

    for sig in signals:
        sig_vals = sig[candidate_slots]
        # Rank with tie-aware averaging: equal values get equal rank.
        # Standard RRF requires this - arbitrary tie-breaking biases
        # toward lower slot indices.
        order = np.argsort(-sig_vals)
        raw_ranks = np.empty(n_cand, dtype=np.float64)
        raw_ranks[order] = np.arange(n_cand, dtype=np.float64)
        # Average ranks for tied values
        sorted_vals = sig_vals[order]
        i = 0
        while i < n_cand:
            j = i + 1
            while j < n_cand and sorted_vals[j] == sorted_vals[i]:
                j += 1
            if j > i + 1:
                avg_rank = np.mean(raw_ranks[order[i:j]])
                raw_ranks[order[i:j]] = avg_rank
            i = j
        # Zero-signal candidates get worst rank (pushed to bottom)
        zero_mask = sig_vals == 0.0
        raw_ranks[zero_mask] = n_cand
        fused[candidate_slots] += 1.0 / (k_rrf + raw_ranks)

    return fused


def normalize_bm25(scores: np.ndarray) -> np.ndarray:
    """Scale BM25 scores by their max. Returns all-zero if max == 0."""
    if scores.size == 0:
        return scores
    m = float(scores.max())
    if m <= 0.0:
        return np.zeros_like(scores, dtype=np.float64)
    return scores.astype(np.float64, copy=False) / m


def recency_decay(
    updated_at_ms: np.ndarray,
    present: np.ndarray,
    now_ms: int,
    half_life_days: float = 30.0,
) -> np.ndarray:
    """Exponential decay score from timestamp column.

    Missing timestamps return 1.0 (treat as most recent).
    """
    if updated_at_ms.size == 0:
        return np.ones(0, dtype=np.float64)
    age_ms = (now_ms - updated_at_ms.astype(np.float64))
    age_days = np.maximum(age_ms / 86400000.0, 0.0)
    decay = np.exp(-age_days / half_life_days)
    return np.where(present, decay, 1.0)


def temporal_proximity(
    event_at_ms: np.ndarray,
    present: np.ndarray,
    anchor_ms: int,
    decay_days: float = 365.0,
) -> np.ndarray:
    """Gaussian proximity score: how close is __event_at__ to the query anchor.

    Returns 1.0 for exact match, decays symmetrically with distance.
    Missing __event_at__ returns 1.0 (neutral - don't penalize non-temporal nodes).
    """
    if event_at_ms.size == 0:
        return np.ones(0, dtype=np.float64)
    dist_ms = np.abs(anchor_ms - event_at_ms.astype(np.float64))
    dist_days = dist_ms / 86400000.0
    sigma = max(decay_days, 1.0)
    score = np.exp(-0.5 * (dist_days / sigma) ** 2)
    return np.where(present, score, 1.0)


def weighted_remember_fusion(
    vec_signal: np.ndarray,
    bm25_signal: np.ndarray,
    recency_signal: np.ndarray,
    confidence_signal: np.ndarray,
    recall_signal: np.ndarray,
    weights: list[float],
) -> np.ndarray:
    """5-signal weighted sum over aligned candidate arrays.

    Returns a float64 array of fused scores indexed the same as the inputs.
    Missing weights fall back to defaults [0.30, 0.20, 0.15, 0.20, 0.15].
    """
    defaults = (0.30, 0.20, 0.15, 0.20, 0.15)
    w = [
        weights[i] if i < len(weights) else defaults[i]
        for i in range(5)
    ]
    return (
        w[0] * vec_signal
        + w[1] * bm25_signal
        + w[2] * recency_signal
        + w[3] * confidence_signal
        + w[4] * recall_signal
    )
