"""Single source of truth for what supergraph/algos/ may import.

Consumed by:
    - tests/test_algos_purity.py  → blocks forbidden imports at CI time
    - benchmarks/algos/dump_env.py → produces autoresearch prompt context
    - benchmarks/algos/README.md   → documented allowlist

Adding a package here is a two-way door: purity test immediately allows
it, and dump_env will surface it to any autoresearch LLM that reads the
environment manifest.
"""

from __future__ import annotations

STDLIB: frozenset[str] = frozenset({
    "math",
    "heapq",
    "collections",
    "dataclasses",
    "typing",
    "re",
    "itertools",
    "functools",
    "operator",
    "enum",
    "bisect",
    "array",
    "struct",
    "abc",
    "contextlib",
})

CORE: dict[str, dict[str, str]] = {
    "numpy": {
        "pypi": "numpy",
        "hint": "array computing primitive. everything vectorized goes through numpy.",
    },
    "scipy": {
        "pypi": "scipy",
        "hint": "sparse matrices (scipy.sparse), csgraph (bfs/dfs/dijkstra/connected_components), spatial, stats.",
    },
}

OPTIONAL: dict[str, dict[str, str]] = {
    "networkx": {
        "pypi": "networkx",
        "hint": "graph algorithms library. pagerank, betweenness, k-shortest-paths, community detection. interops with scipy.sparse.",
    },
    "simsimd": {
        "pypi": "simsimd",
        "hint": "SIMD-accelerated vector ops - cosine, dot, euclidean, hamming. typically 5-10x faster than numpy for small vectors. used by usearch internally.",
    },
    "mmh3": {
        "pypi": "mmh3",
        "hint": "MurmurHash3. fast non-cryptographic hash for bloom filters, locality-sensitive hashing, string keys.",
    },
    "tokenizers": {
        "pypi": "tokenizers",
        "hint": "Rust-backed tokenization from HuggingFace. relevant to text.py for fast word splitting and BPE.",
    },
    "py_rust_stemmers": {
        "pypi": "py_rust_stemmers",
        "hint": "Rust snowball stemmer. relevant to text.py for stemming before fts5 sanitize.",
    },
    "numba": {
        "pypi": "numba",
        "hint": "JIT compiler for numpy-heavy Python. @njit or @vectorize decorators. great for tight loops that numpy can't vectorize.",
    },
    "numexpr": {
        "pypi": "numexpr",
        "hint": "fast numerical expression evaluator. ne.evaluate('a*b+c') beats numpy for memory-bound array ops.",
    },
    "bottleneck": {
        "pypi": "Bottleneck",
        "hint": "fast replacements for numpy.nan* and moving-window ops.",
    },
    "pyarrow": {
        "pypi": "pyarrow",
        "hint": "columnar buffers with zero-copy slicing. relevant if algos need to produce Arrow output.",
    },
}

FORBIDDEN_PREFIXES: frozenset[str] = frozenset({
    "supergraph",
    "pytest",
    "fastapi",
    "uvicorn",
    "starlette",
    "sqlite3",
    "httpx",
    "httpcore",
    "urllib3",
    "urllib",
    "http",
    "logging",
    "asyncio",
    "multiprocessing",
    "subprocess",
    "socket",
    "ssl",
    "threading",
    "os",
    "sys",
    "pathlib",
    "tempfile",
    "shutil",
    "pickle",
    "marshal",
    "shelve",
})


def allowed_import_names() -> frozenset[str]:
    """Every module name that may appear in an algos/ file's imports."""
    return frozenset(STDLIB) | frozenset(CORE.keys()) | frozenset(OPTIONAL.keys())


def is_forbidden(module: str) -> bool:
    """True if import is on the forbidden list."""
    top = module.split(".")[0]
    return top in FORBIDDEN_PREFIXES
