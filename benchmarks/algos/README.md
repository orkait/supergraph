# Algo benchmark harness

Metricable micro-benchmarks for every function in `supergraph/algos/`.
Built on `pytest-benchmark`. Reproducible (fixed seeds), parametrized by
input size, grouped by function, saved to `.benchmarks/` for comparison.

## Why this exists

`supergraph/algos/` holds pure primitives (BFS, dijkstra, compaction remap,
BM25 normalize, 5-signal fusion, spreading activation, FTS5 sanitize).
They are the tunable hot paths. This harness gives us:

- Objective numbers - mean, p50, p95, stddev, ops/sec, rounds
- Comparison - `run.sh compare` shows relative change vs saved baseline
- Regression gate - `run.sh gate` fails CI if any algo is ≥5% slower
- No SuperGraph fixture needed - benchmarks call algos directly with
  synthetic numpy inputs

## Scope

Not covered by this harness (intentionally):

- End-to-end DSL latency - use the `longmemeval` bench framework for that
- Embedder throughput - that's dominated by the model, not our code
- SQLite / usearch internals - external libraries, out of scope
- Network / disk I/O - no I/O in the algos layer

## Running

```bash
# From repo root
./benchmarks/algos/run.sh run        # run all, autosave result
./benchmarks/algos/run.sh baseline   # save current run as 'baseline'
./benchmarks/algos/run.sh compare    # diff current against baseline, warn at >10%
./benchmarks/algos/run.sh gate       # strict gate, fail at >5%
./benchmarks/algos/run.sh list       # show saved runs
```

Pass pytest filters as extra args:

```bash
./benchmarks/algos/run.sh run -k "bfs or dijkstra"
./benchmarks/algos/run.sh run benchmarks/algos/test_fusion_bench.py
```

## Typical tuning workflow

```bash
# 1. Capture reference before changing anything
./benchmarks/algos/run.sh baseline

# 2. Edit supergraph/algos/<file>.py - tune the algorithm

# 3. See what your change did
./benchmarks/algos/run.sh compare

#    Columns:
#    - Name    the benchmark
#    - Min     best case observed
#    - Mean    average (primary metric)
#    - Max     worst case
#    - StdDev  noise
#    - Rounds  number of runs pytest-benchmark did

# 4. If it's an improvement, lock it as the new baseline
./benchmarks/algos/run.sh baseline

# 5. If it's a regression, revert and try again
```

## Fixture sizes

Graphs (random directed, ~10 out-edges per node):
- `graph_1k`      - 1 000 nodes
- `graph_10k`     - 10 000 nodes
- `graph_100k`    - 100 000 nodes
- `graph_1k_dense` - 1 000 nodes, ~50 out-edges (for path enumeration stress)

Edge lists (for compact remap):
- `edge_list_10k`  - 30 000 edges over 10 k nodes
- `edge_list_100k` - 500 000 edges over 100 k nodes

Fusion signal arrays:
- `fusion_signals_1k`  - 5 × 1 000 random floats
- `fusion_signals_10k` - 5 × 10 000 random floats

Live masks:
- `live_mask_10k_80pct`, `live_mask_100k_80pct` - 80% live, 20% tombstoned

Every fixture uses a fixed seed. If the seed changes, baselines become
invalid - treat seeds as part of the harness contract.

## What's covered

| Module | Fns benched | Sizes |
|---|---|---|
| `graph` | bfs_traverse, dijkstra, bidirectional_bfs, find_all_paths, common_neighbors | 1k, 10k, 100k + dense 1k |
| `compact` | build_live_mask, slot_remap_plan, apply_slot_remap_to_edges | 10k, 100k |
| `fusion` | rrf_fuse, normalize_bm25, recency_decay, weighted_remember_fusion | 100, 1k, 10k |
| `spreading` | spreading_activation | 1k, 10k, 100k × depth 2-4 |
| `eviction` | needs_optimization, rank_evictable_slots | 10k |
| `text` | fts5_sanitize, tokenize_unicode | short, long, batch |

## Interpretation guide

- **Mean** is the primary metric. Sort by it with `--benchmark-sort=mean`.
- **StdDev / mean > 20%** means the benchmark is noisy - pin CPU, disable
  turbo, close other processes, or increase `--benchmark-min-rounds`.
- **Min** is useful for detecting best-case theoretical speed.
- **Max** matters if you care about tail latency (usually you don't for
  a batch algo).

Perf wins to watch for when tuning:

- `bfs_traverse_100k_depth3` - scales with nnz × depth. Matrix-power is
  O(nnz × depth). Python BFS would be ~5-20× slower here.
- `dijkstra_100k` - scipy native C. Python heap version was ~50× slower.
- `apply_slot_remap_to_edges_100k` - dominated by Python list
  construction. If this becomes a bottleneck, consider a numpy struct array.
- `weighted_remember_fusion_10k` - pure numpy vectorized. Should be tens
  of microseconds, not milliseconds. If slower, someone added a Python loop.

## Regression gate

`run.sh gate` is the CI-friendly form. Exits non-zero if any benchmark is
≥5% slower than the saved `baseline` run. Wire into CI after a perf work
stream lands to prevent silent regressions.

## Autoresearch workflow (Karpathy-style)

For tools like [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
that iteratively rewrite ONE target file and need a runnable metric command:

```bash
# File under improvement + multi-metric command
python -m benchmarks.algos.bench_one graph --fast --quiet
```

Output format (stdout, last N lines):
```
METRIC_FILE supergraph/algos/graph.py
METRIC TestBfsTraverse::test_1k_depth2 51.4514
METRIC TestBfsTraverse::test_10k_depth2 279.4876
METRIC TestDijkstra::test_100k 39486.5829
...
```

Or pure JSON for programmatic parsing:

```bash
python -m benchmarks.algos.bench_one graph --fast --json
```

```json
{
  "algo": "graph",
  "file_under_improvement": "supergraph/algos/graph.py",
  "unit": "microseconds",
  "lower_is_better": true,
  "metrics": {
    "TestBfsTraverse::test_1k_depth2": 51.45,
    "TestBfsTraverse::test_10k_depth2": 279.48,
    "TestDijkstra::test_100k": 39486.58,
    ...
  }
}
```

### Autoresearch config example

```json
{
  "file_under_improvement": "supergraph/algos/graph.py",
  "metric_command": "python -m benchmarks.algos.bench_one graph --fast --json",
  "metric_format": "json",
  "metric_direction": "lower_is_better",
  "environment_manifest": "python -m benchmarks.algos.dump_env",
  "safety_command": "pytest tests/test_algos_purity.py tests/test_dsl_user_reads.py"
}
```

### Environment manifest - `dump_env.py`

The LLM needs to know **exact installed packages and versions** so it can
pick real libraries (not imagined ones). `dump_env` reads the canonical
allowlist at `benchmarks/algos/allowlist.py`, cross-references with the
live Python environment, and outputs the intersection.

```bash
# Markdown context for LLM prompt injection
python -m benchmarks.algos.dump_env

# Machine-parseable JSON
python -m benchmarks.algos.dump_env --json

# One-liner requirement style
python -m benchmarks.algos.dump_env --compact

# Write to a file autoresearch can reference
python -m benchmarks.algos.dump_env --write benchmarks/algos/ENVIRONMENT.md
```

Example output:

```
# python 3.10.20
# core (required):
numpy==2.2.6  # import numpy
scipy==1.15.3  # import scipy
# optional (installed):
networkx==3.4.2  # import networkx
simsimd==6.5.16  # import simsimd
mmh3==5.2.1  # import mmh3
tokenizers==0.22.2  # import tokenizers
py_rust_stemmers==0.1.5  # import py_rust_stemmers
```

When prompting the LLM for a rewrite, include the markdown manifest
(`ENVIRONMENT.md`) so it knows what's available. The purity gate
enforces the allowlist - if the LLM imports something not listed,
`pytest tests/test_algos_purity.py` fails and autoresearch reverts.

### Adding a new package to the allowlist

1. Edit `benchmarks/algos/allowlist.py` - add the entry under `OPTIONAL`
   with `pypi` name and `hint`.
2. Install the package: `uv pip install <pypi-name>`
3. Regenerate `ENVIRONMENT.md`: `python -m benchmarks.algos.dump_env --write benchmarks/algos/ENVIRONMENT.md`
4. Run `pytest tests/test_algos_purity.py` to confirm the allowlist
   updates are picked up.
5. Commit allowlist.py + ENVIRONMENT.md together.

### One-file-at-a-time: every algo is independently optimizable

| Algo          | File to improve                 | Command                                                       |
|---------------|--------------------------------|---------------------------------------------------------------|
| graph         | `supergraph/algos/graph.py`    | `python -m benchmarks.algos.bench_one graph --fast --json`    |
| compact       | `supergraph/algos/compact.py`  | `python -m benchmarks.algos.bench_one compact --fast --json`  |
| fusion        | `supergraph/algos/fusion.py`   | `python -m benchmarks.algos.bench_one fusion --fast --json`   |
| spreading     | `supergraph/algos/spreading.py`| `python -m benchmarks.algos.bench_one spreading --fast --json`|
| eviction      | `supergraph/algos/eviction.py` | `python -m benchmarks.algos.bench_one eviction --fast --json` |
| text          | `supergraph/algos/text.py`     | `python -m benchmarks.algos.bench_one text --fast --json`     |

Each command:
- Runs only that algo's benchmark file
- Emits one metric per benchmark (many metrics, one file)
- Writes structured JSON to `.benchmarks/metric_<algo>_summary.json`
- Exits non-zero if the bench or any import fails (autoresearch reverts)

### Safety net

Autoresearch may propose edits that break the algo's contract. Layered
guards catch this:

1. **Purity gate** - `tests/test_algos_purity.py` fails if the rewrite
   introduces a forbidden import (anything from `supergraph.*`).
2. **Behavioural regression** - `tests/test_dsl_user_reads.py` exercises
   traversal / paths / recall / similar through the real stack. Any
   semantic regression shows up here.
3. **Bench crash** - if the proposed rewrite has a runtime error, the
   bench run fails (non-zero exit), autoresearch sees failure, reverts.
4. **Benchmark noise** - `--fast` mode keeps each benchmark ~0.3s, so
   stddev is higher; use multiple runs or `--fast` off for final
   verification.

Recommended autoresearch loop:

```
1. bench_one <algo> --fast --json          # fast iterate on metrics
2. pytest tests/test_algos_purity.py       # purity guard
3. pytest tests/test_dsl_user_reads.py     # behavior guard
4. accept edit iff all three pass AND metric improved
```

## Adding a new benchmark

1. Add the algo to `supergraph/algos/<file>.py`
2. Add fixtures (if needed) to `benchmarks/algos/conftest.py`
3. Add a `TestXxx::test_...` to `benchmarks/algos/test_<file>_bench.py`
4. Run `./benchmarks/algos/run.sh baseline` to capture initial number
5. Commit the code + a one-line update to this README listing the new case
