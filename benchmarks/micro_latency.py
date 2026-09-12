from __future__ import annotations

import statistics
import sys
import tempfile
import time

from supergraph import SuperGraph


def _bulk_create(gs: SuperGraph, n: int) -> float:
    t0 = time.perf_counter()
    with gs.deferred_embeddings(batch_size=256):
        for i in range(n):
            gs.execute(
                f'CREATE NODE "n{i}" kind = "memory" '
                f'DOCUMENT "Record about topic {i % 100}. Body {i}."'
            )
    return time.perf_counter() - t0


def _bench(gs: SuperGraph, q: str, iters: int = 30, warmup: int = 5) -> float:
    for _ in range(warmup):
        gs.execute(q)
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        gs.execute(q)
        times.append((time.perf_counter() - t) * 1e6)
    return statistics.median(times)


QUERIES: list[tuple[str, str]] = [
    ("point lookup   NODE",                'NODE "n42"'),
    ("filtered scan  NODES LIMIT 10",      'NODES WHERE kind = "memory" LIMIT 10'),
    ("SIMILAR TO     LIMIT 10",            'SIMILAR TO "topic description" LIMIT 10'),
    ("RECALL DEPTH 3 LIMIT 10",            'RECALL FROM "n0" DEPTH 3 LIMIT 10'),
    ("REMEMBER       LIMIT 10",            'REMEMBER "topic description" LIMIT 10'),
    ("ASSERT",                             'ASSERT "f1" kind = "fact" value = 1 CONFIDENCE 0.9 SOURCE "t"'),
]


def run(n: int, mode: str) -> None:
    if mode == "mem":
        gs = SuperGraph(path=None)
    else:
        td = tempfile.mkdtemp(prefix=f"gs_bench_{n}_")
        gs = SuperGraph(path=f"{td}/db")

    ingest_s = _bulk_create(gs, n)

    for i in range(min(2000, n - 1)):
        gs.execute(f'CREATE EDGE "n{i}" -> "n{i+1}" kind = "next"')

    print(f"\n=== N={n:,} {mode}  bulk-create {ingest_s:.1f}s ===")
    for label, q in QUERIES:
        us = _bench(gs, q, iters=(10 if "ASSERT" in label else 30))
        print(f"  {label:<32} {us:>10.1f} us")
    gs.close()


def main() -> None:
    sizes = [int(x) for x in (sys.argv[1:] if len(sys.argv) > 1 else ["10000", "100000"])]
    for n in sizes:
        for mode in ("mem", "disk"):
            run(n, mode)


if __name__ == "__main__":
    main()
