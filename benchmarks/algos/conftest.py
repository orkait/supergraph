
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix


SEED = 42


def _rng(seed_offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED + seed_offset)


def _make_random_graph(n: int, avg_degree: int, seed_offset: int = 0) -> csr_matrix:
    rng = _rng(seed_offset)
    total_edges = n * avg_degree
    sources = rng.integers(0, n, total_edges, dtype=np.int32)
    targets = rng.integers(0, n, total_edges, dtype=np.int32)
    weights = rng.random(total_edges, dtype=np.float32) + 0.1
    return csr_matrix((weights, (sources, targets)), shape=(n, n))


@pytest.fixture(scope="session")
def graph_1k() -> csr_matrix:
    return _make_random_graph(1_000, 10)


@pytest.fixture(scope="session")
def graph_10k() -> csr_matrix:
    return _make_random_graph(10_000, 10)


@pytest.fixture(scope="session")
def graph_100k() -> csr_matrix:
    return _make_random_graph(100_000, 10)


@pytest.fixture(scope="session")
def graph_1k_dense() -> csr_matrix:
    return _make_random_graph(1_000, 50)


@pytest.fixture(scope="session")
def edge_list_10k() -> list[tuple[int, int, dict]]:
    rng = _rng(seed_offset=1)
    n = 10_000
    m = 30_000
    srcs = rng.integers(0, n, m, dtype=np.int64).tolist()
    tgts = rng.integers(0, n, m, dtype=np.int64).tolist()
    return [(s, t, {"w": 1.0}) for s, t in zip(srcs, tgts)]


@pytest.fixture(scope="session")
def edge_list_100k() -> list[tuple[int, int, dict]]:
    rng = _rng(seed_offset=2)
    n = 100_000
    m = 500_000
    srcs = rng.integers(0, n, m, dtype=np.int64).tolist()
    tgts = rng.integers(0, n, m, dtype=np.int64).tolist()
    return [(s, t, {"w": 1.0}) for s, t in zip(srcs, tgts)]


@pytest.fixture(scope="session")
def live_mask_10k_80pct() -> np.ndarray:
    rng = _rng(seed_offset=3)
    mask = rng.random(10_000) < 0.80
    return mask


@pytest.fixture(scope="session")
def live_mask_100k_80pct() -> np.ndarray:
    rng = _rng(seed_offset=4)
    mask = rng.random(100_000) < 0.80
    return mask


@pytest.fixture(scope="session")
def node_ids_10k() -> np.ndarray:
    rng = _rng(seed_offset=5)
    return rng.integers(0, 10_000, 10_000, dtype=np.int32)


@pytest.fixture(scope="session")
def node_ids_100k() -> np.ndarray:
    rng = _rng(seed_offset=6)
    return rng.integers(0, 100_000, 100_000, dtype=np.int32)


@pytest.fixture(scope="session")
def fusion_signals_1k() -> dict:
    rng = _rng(seed_offset=10)
    m = 1_000
    return {
        "vec": rng.random(m, dtype=np.float64),
        "bm25": rng.random(m, dtype=np.float64),
        "recency": rng.random(m, dtype=np.float64),
        "confidence": rng.random(m, dtype=np.float64),
        "recall": rng.random(m, dtype=np.float64),
    }


@pytest.fixture(scope="session")
def fusion_signals_10k() -> dict:
    rng = _rng(seed_offset=11)
    m = 10_000
    return {
        "vec": rng.random(m, dtype=np.float64),
        "bm25": rng.random(m, dtype=np.float64),
        "recency": rng.random(m, dtype=np.float64),
        "confidence": rng.random(m, dtype=np.float64),
        "recall": rng.random(m, dtype=np.float64),
    }


@pytest.fixture(scope="session")
def bm25_raw_10k() -> np.ndarray:
    rng = _rng(seed_offset=12)
    return rng.exponential(scale=2.0, size=10_000).astype(np.float64)


@pytest.fixture(scope="session")
def updated_at_10k() -> tuple[np.ndarray, np.ndarray]:
    rng = _rng(seed_offset=13)
    now_ms = 1_700_000_000_000
    ages_ms = rng.integers(0, 365 * 86_400_000, 10_000, dtype=np.int64)
    timestamps = now_ms - ages_ms
    present = rng.random(10_000) < 0.9
    return timestamps, present


@pytest.fixture(scope="session")
def updated_at_100k() -> tuple[np.ndarray, np.ndarray]:
    rng = _rng(seed_offset=14)
    now_ms = 1_700_000_000_000
    ages_ms = rng.integers(0, 365 * 86_400_000, 100_000, dtype=np.int64)
    timestamps = now_ms - ages_ms
    present = rng.random(100_000) < 0.9
    return timestamps, present


@pytest.fixture(scope="session")
def eviction_inputs_10k(
    live_mask_10k_80pct: np.ndarray,
    node_ids_10k: np.ndarray,
    updated_at_10k: tuple[np.ndarray, np.ndarray],
) -> dict:
    kinds = ["user", "document", "chunk", "message", "schema", "config"]
    return {
        "live_mask": live_mask_10k_80pct,
        "kind_ids": node_ids_10k % len(kinds),
        "kind_lookup": lambda i: kinds[i],
        "updated_at": updated_at_10k[0],
        "updated_at_present": updated_at_10k[1],
        "protected_kinds": {"schema", "config"},
    }


@pytest.fixture(scope="session")
def eviction_inputs_100k(
    live_mask_100k_80pct: np.ndarray,
    node_ids_100k: np.ndarray,
    updated_at_100k: tuple[np.ndarray, np.ndarray],
) -> dict:
    kinds = ["user", "document", "chunk", "message", "schema", "config"]
    return {
        "live_mask": live_mask_100k_80pct,
        "kind_ids": node_ids_100k % len(kinds),
        "kind_lookup": lambda i: kinds[i],
        "updated_at": updated_at_100k[0],
        "updated_at_present": updated_at_100k[1],
        "protected_kinds": {"schema", "config"},
    }


@pytest.fixture(scope="session")
def fts5_queries() -> list[str]:
    return [
        "what is the user's favorite food?",
        "when did we last discuss the project deadline",
        "find all references to Kubernetes and Docker in recent messages",
        "AND OR NOT NEAR special keywords mixed in a sentence",
        "a",
        "the quick brown fox jumps over the lazy dog",
        "user_id = 42 AND kind = 'conversation'",
        "empty tokens ? ! * + - ~",
    ]
