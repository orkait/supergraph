import numpy as np

from supergraph.algos.fusion import (
    normalize_bm25,
    recency_decay,
    rrf_fuse,
    weighted_remember_fusion,
)


def test_rrf_fuse_sums_scores_across_groups():
    result = rrf_fuse(
        [
            {"a": 0, "b": 1},
            {"a": 2, "c": 0},
        ],
        k_rrf=60.0,
    )
    assert np.isclose(result["a"], (1.0 / 60.0) + (1.0 / 62.0))
    assert np.isclose(result["b"], 1.0 / 61.0)
    assert np.isclose(result["c"], 1.0 / 60.0)


def test_normalize_bm25_scales_by_max():
    scores = np.array([0.0, 2.0, 4.0], dtype=np.float64)
    normalized = normalize_bm25(scores)
    assert np.allclose(normalized, np.array([0.0, 0.5, 1.0]))


def test_normalize_bm25_returns_zeros_when_max_is_non_positive():
    scores = np.array([-1.0, 0.0, -3.0], dtype=np.float64)
    normalized = normalize_bm25(scores)
    assert np.array_equal(normalized, np.zeros_like(scores))


def test_recency_decay_uses_one_for_missing_values():
    updated = np.array([1_700_000_000_000, 1_699_000_000_000], dtype=np.int64)
    present = np.array([True, False], dtype=bool)
    decay = recency_decay(updated, present, now_ms=1_700_000_000_000, half_life_days=30.0)
    assert np.isclose(decay[0], 1.0)
    assert np.isclose(decay[1], 1.0)


def test_weighted_remember_fusion_uses_defaults_for_missing_weights():
    vec = np.array([1.0], dtype=np.float64)
    bm25 = np.array([2.0], dtype=np.float64)
    recency = np.array([3.0], dtype=np.float64)
    confidence = np.array([4.0], dtype=np.float64)
    recall = np.array([5.0], dtype=np.float64)
    fused = weighted_remember_fusion(vec, bm25, recency, confidence, recall, [1.0, 2.0])
    expected = (1.0 * vec) + (2.0 * bm25) + (0.15 * recency) + (0.20 * confidence) + (0.15 * recall)
    assert np.allclose(fused, expected)
