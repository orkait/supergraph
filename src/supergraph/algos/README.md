# supergraph.algos

Pure algorithmic primitives. Tunable in isolation, benchmarkable without a
SuperGraph instance, callable from anywhere in the codebase.

## Contract (non-negotiable)

1. **No imports from `supergraph.*`** (any submodule). Allowed:
   - Python stdlib (`math`, `heapq`, `collections`, `dataclasses`, `typing`, `re`)
   - `numpy`
   - `scipy.sparse`, `scipy.sparse.csgraph`

2. **No I/O.** No SQLite, no files, no network, no logging.

3. **No global mutable state.** Functions are pure or explicitly in-place on
   their arguments.

4. **Determinism.** Same input → same output. No wall-clock reads inside
   algos - caller passes `now_ms` if needed.

5. **Explicit `__all__`.** Each module declares its public surface.

6. **Type-hinted public API.**

## Modules

| Module | Exports | Used by |
|---|---|---|
| `graph.py` | `bfs_traverse`, `dijkstra`, `bidirectional_bfs`, `find_all_paths`, `common_neighbors` | `core/path.py`, `dsl/handlers/traversal.py` |
| `compact.py` | `build_live_mask`, `slot_remap_plan`, `apply_slot_remap_to_edges` | `core/optimizer.py::compact_tombstones` |
| `eviction.py` | `needs_optimization`, `rank_evictable_slots` | `core/optimizer.py` |
| `fusion.py` | `rrf_fuse`, `normalize_bm25`, `recency_decay`, `weighted_remember_fusion` | `dsl/handlers/intelligence.py::_remember` |
| `spreading.py` | `spreading_activation` | `dsl/handlers/intelligence.py::_recall` |
| `text.py` | `fts5_sanitize`, `tokenize_unicode` | `document/store.py` |

## Design pattern - Functional Core / Imperative Shell

- **Pure core** = this package. Computes what should happen.
- **Imperative shell** = `core/store.py`, `dsl/handlers/*`, `core/optimizer.py`.
  Calls the pure core, applies the result to mutable state.

This means algorithmic correctness is testable with synthetic numpy arrays,
without booting a SuperGraph. Performance tuning can be driven by
algo-level benchmarks. Algorithms can be reused across adapters, benches,
and handlers without dragging state coupling along.
