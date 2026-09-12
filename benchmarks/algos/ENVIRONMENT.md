# supergraph/algos - environment manifest

- python: **3.10.20**
- platform: Linux-6.17.0-20-generic-x86_64-with-glibc2.39

> This file lists the **exact packages and versions**
> available when rewriting a file in `supergraph/algos/`.
> Do not import anything not listed here - the purity
> gate will reject the patch.

## Core (always available, always allowed)

- **`numpy==2.2.6`** - `import numpy`
  - array computing primitive. everything vectorized goes through numpy.
- **`scipy==1.15.3`** - `import scipy`
  - sparse matrices (scipy.sparse), csgraph (bfs/dfs/dijkstra/connected_components), spatial, stats.

## Optional (installed - you may use any of these)

- **`networkx==3.4.2`** - `import networkx`
  - graph algorithms library. pagerank, betweenness, k-shortest-paths, community detection. interops with scipy.sparse.
- **`simsimd==6.5.16`** - `import simsimd`
  - SIMD-accelerated vector ops - cosine, dot, euclidean, hamming. typically 5-10x faster than numpy for small vectors. used by usearch internally.
- **`mmh3==5.2.1`** - `import mmh3`
  - MurmurHash3. fast non-cryptographic hash for bloom filters, locality-sensitive hashing, string keys.
- **`tokenizers==0.22.2`** - `import tokenizers`
  - Rust-backed tokenization from HuggingFace. relevant to text.py for fast word splitting and BPE.
- **`py_rust_stemmers==0.1.5`** - `import py_rust_stemmers`
  - Rust snowball stemmer. relevant to text.py for stemming before fts5 sanitize.

## Optional (NOT installed - do NOT import)

- `numba` - not available in this environment
- `numexpr` - not available in this environment
- `Bottleneck` - not available in this environment
- `pyarrow` - not available in this environment

## Python standard library (allowed)

`abc`, `array`, `bisect`, `collections`, `contextlib`, `dataclasses`, `enum`, `functools`, `heapq`, `itertools`, `math`, `operator`, `re`, `struct`, `typing`

## Forbidden imports (purity gate rejects these)

`asyncio`, `fastapi`, `supergraph`, `http`, `httpcore`, `httpx`, `logging`, `marshal`, `multiprocessing`, `os`, `pathlib`, `pickle`, `pytest`, `shelve`, `shutil`, `socket`, `sqlite3`, `ssl`, `starlette`, `subprocess`, `sys`, `tempfile`, `threading`, `urllib`, `urllib3`, `uvicorn`

## Rules

1. Import only from the lists above
2. No I/O (files, sockets, SQLite, network)
3. No logging or global state
4. Pure functions - do not mutate caller state unless documented
5. Deterministic - do not read wall-clock time inside algos
6. Type-hint the public API
7. Preserve function signatures so existing callers keep working
