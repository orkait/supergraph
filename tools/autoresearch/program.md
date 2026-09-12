# Algo Optimization Instructions

## Goal
Minimize **mean latency (microseconds)** across all benchmark cases. Lower is better. Every benchmark must still pass.

## OUTPUT FORMAT (STRICT)
Your entire response MUST be a valid Python source file - nothing else.

- DO NOT include any explanation, commentary, or preamble before the code.
- DO NOT include any summary, notes, or disclaimers after the code.
- DO NOT wrap the code in ```python ... ``` fences.
- DO NOT add "Here is the optimized version:" or similar framing.
- The first character of your response MUST be a valid Python token
  (`from`, `import`, `def`, `class`, `@`, `"""`, `#`).
- The last character MUST be the final character of Python source.

If you include ANY prose outside the Python source, the response is rejected
and the optimization attempt is wasted. Output raw code only.

## Hard constraints (purity gate - violations are auto-rejected)

### Allowed imports
Only import from this list:

**Core (always available):**
- `numpy` - vectorized array ops
- `scipy` - sparse matrices (scipy.sparse), csgraph (bfs/dfs/dijkstra/connected_components)

**Optional (installed):**
- `networkx` - pagerank, betweenness, k-shortest-paths, community detection
- `simsimd` - SIMD vector ops (cosine, dot, euclidean, hamming) - 5-10x faster than numpy for small vectors
- `mmh3` - MurmurHash3, fast non-cryptographic hash
- `tokenizers` - Rust-backed tokenization (HuggingFace)
- `py_rust_stemmers` - Rust snowball stemmer

**Standard library (allowed subset):**
`math`, `heapq`, `collections`, `dataclasses`, `typing`, `re`, `itertools`,
`functools`, `operator`, `enum`, `bisect`, `array`, `struct`, `abc`, `contextlib`

### Forbidden (auto-rejected if found)
`supergraph`, `pytest`, `fastapi`, `uvicorn`, `starlette`, `sqlite3`, `httpx`,
`httpcore`, `urllib3`, `urllib`, `http`, `logging`, `asyncio`, `multiprocessing`,
`subprocess`, `socket`, `ssl`, `threading`, `os`, `sys`, `pathlib`, `tempfile`,
`shutil`, `pickle`, `marshal`, `shelve`

## Rules
1. **Never change function signatures** - callers must keep working
2. **No I/O, no side effects, no global mutable state**
3. **Deterministic** - same inputs always produce same outputs
4. **At least as correct as the original** - benchmarks assert on return values

## Optimization ideas
- no loops → numpy ufuncs/where/nonzero
- graph traversal → scipy.sparse.csgraph
- set ops → np.intersect1d/union1d
- queues → collections.deque
- pre-alloc arrays, no append
- visited → np.bool_ array
- frontier → bitwise
- priority → heapq tuples
- sparse → CSR indptr/indices direct, no .toarray()
