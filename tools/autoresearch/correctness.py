
from __future__ import annotations

import types
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, spmatrix


SEED = 42


def _rng(off: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED + off)


def _random_graph(n: int, avg_degree: int, off: int = 0) -> csr_matrix:
    rng = _rng(off)
    total = n * avg_degree
    s = rng.integers(0, n, total, dtype=np.int32)
    t = rng.integers(0, n, total, dtype=np.int32)
    w = rng.random(total, dtype=np.float32) + 0.1
    return csr_matrix((w, (s, t)), shape=(n, n))


def _inputs_spreading() -> list[tuple[str, tuple, dict]]:
    g = _random_graph(1_000, 10)
    g_t = g.T.tocsr()
    live = np.ones(1_000, dtype=bool)
    imp = _rng(7).random(1_000).astype(np.float64)
    rec = _rng(8).random(1_000).astype(np.float64)
    return [
        ("spreading_activation", (g_t, 0, 2, 0.7, live, None, None), {}),
        ("spreading_activation", (g_t, 0, 4, 0.7, live, None, None), {}),
        ("spreading_activation", (g_t, 0, 3, 0.7, live, imp, rec), {}),
    ]


def _inputs_graph() -> list[tuple[str, tuple, dict]]:
    g_sparse = _random_graph(1_000, 10)
    g_dense = _random_graph(1_000, 50)
    g_t = g_sparse.T.tocsr()
    return [
        ("bfs_traverse", (g_sparse, 0, 2), {}),
        ("bfs_traverse", (g_sparse, 0, 5), {}),
        ("dijkstra", (g_sparse, 0, 500), {}),
        ("bidirectional_bfs", (g_sparse, g_t, 0, 500, 10), {}),
        ("find_all_paths", (g_dense, 0, 500, 3), {}),
        ("common_neighbors", (g_sparse, 0, 1), {}),
    ]


def _inputs_fusion() -> list[tuple[str, tuple, dict]]:
    g1 = {f"id_{i}": i for i in range(100)}
    g2 = {f"id_{i}": i for i in range(100)}
    g3 = {f"id_{i}": i for i in range(100)}
    bm25 = _rng(12).exponential(scale=2.0, size=1_000).astype(np.float64)
    ts = 1_700_000_000_000 - _rng(13).integers(0, 365 * 86_400_000, 1_000, dtype=np.int64)
    pres = _rng(13).random(1_000) < 0.9

    r = _rng(10)
    signals = {
        "vec":        r.random(1_000, dtype=np.float64),
        "bm25":       r.random(1_000, dtype=np.float64),
        "recency":    r.random(1_000, dtype=np.float64),
        "confidence": r.random(1_000, dtype=np.float64),
        "recall":     r.random(1_000, dtype=np.float64),
    }
    return [
        ("rrf_fuse", ([g1, g2],), {}),
        ("rrf_fuse", ([g1, g2, g3],), {}),
        ("normalize_bm25", (bm25,), {}),
        ("recency_decay", (ts, pres, 1_700_000_000_000, 30.0), {}),
        ("weighted_remember_fusion",
         (signals["vec"], signals["bm25"], signals["recency"],
          signals["confidence"], signals["recall"],
          [0.30, 0.20, 0.15, 0.20, 0.15]), {}),
    ]


def _inputs_eviction() -> list[tuple[str, tuple, dict]]:
    health_noop = {
        "tombstone_ratio": 0.0, "string_bloat": 1.0, "live_nodes": 1000,
        "dead_vectors": 0, "stale_edge_keys": 0, "cache_size": 10,
    }
    health_all = {
        "tombstone_ratio": 0.5, "string_bloat": 10.0, "live_nodes": 1000,
        "dead_vectors": 50, "stale_edge_keys": 20, "cache_size": 1000,
    }
    n = 1_000
    live_mask = _rng(3).random(n) < 0.80
    node_ids = _rng(5).integers(0, n, n, dtype=np.int32)
    ages = _rng(13).integers(0, 365 * 86_400_000, n, dtype=np.int64)
    ts = 1_700_000_000_000 - ages
    pres = _rng(13).random(n) < 0.9
    kinds = ["user", "document", "chunk", "message", "schema", "config"]
    eviction_inputs = {
        "live_mask": live_mask,
        "kind_ids": node_ids % len(kinds),
        "kind_lookup": lambda i: kinds[i],
        "updated_at": ts,
        "updated_at_present": pres,
        "protected_kinds": {"schema", "config"},
    }
    return [
        ("needs_optimization", (health_noop,), {}),
        ("needs_optimization", (health_all,), {}),
        ("rank_evictable_slots", (), eviction_inputs),
    ]


def _inputs_compact() -> list[tuple[str, tuple, dict]]:
    n = 1_000
    node_ids = _rng(5).integers(0, n, n, dtype=np.int32)
    tombstones = {int(x) for x in _rng(20).integers(0, n, 100)}
    live_mask = node_ids[:n] >= 0
    mask_arr = live_mask.copy()
    for t in tombstones:
        if t < n:
            mask_arr[t] = False

    new_count = int(mask_arr.sum())
    old_to_new = np.full(n, -1, dtype=np.int32)
    live_slots = np.nonzero(mask_arr)[0]
    old_to_new[live_slots] = np.arange(new_count, dtype=np.int32)

    r = _rng(1)
    m = 3_000
    srcs = r.integers(0, n, m, dtype=np.int64).tolist()
    tgts = r.integers(0, n, m, dtype=np.int64).tolist()
    edge_list = [(s, t, {"w": 1.0}) for s, t in zip(srcs, tgts)]

    return [
        ("build_live_mask", (node_ids, tombstones, n), {}),
        ("slot_remap_plan", (mask_arr,), {}),
        ("apply_slot_remap_to_edges", (edge_list, old_to_new, n), {}),
    ]


def _inputs_text() -> list[tuple[str, tuple, dict]]:
    long_text = ("the quick brown fox jumps over the lazy dog. " * 50).strip()
    short_text = "hello world"
    fts_query = "AND OR NOT keyword * + -"
    cases: list[tuple[str, tuple, dict]] = [
        ("tokenize", (long_text,), {}),
        ("tokenize", (short_text,), {}),
        ("fts5_sanitize", (fts_query,), {}),
        ("fts5_sanitize", (long_text,), {}),
    ]
    return cases


_DISPATCH = {
    "spreading": _inputs_spreading,
    "graph": _inputs_graph,
    "fusion": _inputs_fusion,
    "eviction": _inputs_eviction,
    "compact": _inputs_compact,
    "text": _inputs_text,
}


def _load_module(name: str, code: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__["__name__"] = name
    exec(code, mod.__dict__)
    return mod


def _to_array(x: Any) -> np.ndarray | None:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, spmatrix):
        return x.toarray()
    return None


def _compare(a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    if type(a) is not type(b):
        arr_a = _to_array(a)
        arr_b = _to_array(b)
        if arr_a is not None and arr_b is not None:
            return arr_a.shape == arr_b.shape and np.allclose(arr_a, arr_b, rtol=rtol, atol=atol, equal_nan=True)
        try:
            return a == b
        except Exception:
            return False

    arr_a = _to_array(a)
    if arr_a is not None:
        arr_b = _to_array(b)
        if arr_a.shape != arr_b.shape:
            return False
        return np.allclose(arr_a, arr_b, rtol=rtol, atol=atol, equal_nan=True)

    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_compare(x, y, rtol, atol) for x, y in zip(a, b))

    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_compare(a[k], b[k], rtol, atol) for k in a)

    if isinstance(a, set):
        return a == b

    if isinstance(a, float):
        return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)

    try:
        return a == b
    except Exception:
        return False


def check_correctness(candidate_code: str, baseline_code: str, algo: str) -> str | None:
    if algo not in _DISPATCH:
        return None

    try:
        baseline_mod = _load_module(f"{algo}_baseline_oracle", baseline_code)
    except Exception as e:
        return f"baseline module load failed: {type(e).__name__}: {e}"
    try:
        candidate_mod = _load_module(f"{algo}_candidate_oracle", candidate_code)
    except Exception as e:
        return f"candidate module load failed: {type(e).__name__}: {e}"

    try:
        cases = _DISPATCH[algo]()
    except Exception as e:
        return f"input generation failed: {type(e).__name__}: {e}"

    for fn_name, args, kwargs in cases:
        b_fn = getattr(baseline_mod, fn_name, None)
        c_fn = getattr(candidate_mod, fn_name, None)
        if b_fn is None:
            return f"baseline missing function {fn_name!r}"
        if c_fn is None:
            return f"candidate missing function {fn_name!r}"

        try:
            b_out = b_fn(*args, **kwargs)
        except Exception as e:
            return f"baseline {fn_name!r} raised: {type(e).__name__}: {e}"
        try:
            c_out = c_fn(*args, **kwargs)
        except Exception as e:
            return f"candidate {fn_name!r} raised: {type(e).__name__}: {e}"

        if not _compare(b_out, c_out):
            return f"output drift in {fn_name!r}: candidate differs from baseline"

    return None
