#!/usr/bin/env python3
"""Per-algo benchmark runner for Karpathy-style autoresearch.

One file under improvement → many named metrics. Autoresearch
picks which metric(s) to care about.

Contract:
    - Exit 0 on success
    - Structured JSON at .benchmarks/metric_<algo>_summary.json
    - Stdout prints one line per benchmark:
          METRIC <benchmark_name> <mean_us>
      plus a trailing:
          METRIC_FILE supergraph/algos/<algo>.py
    - All metrics are microseconds, lower is better
    - Raw pytest-benchmark dump at .benchmarks/metric_<algo>.json

Usage:
    python -m benchmarks.algos.bench_one graph
    python -m benchmarks.algos.bench_one graph --fast     # short per-bench budget
    python -m benchmarks.algos.bench_one fusion --quiet   # suppress pytest chatter
    python -m benchmarks.algos.bench_one graph --json     # stdout becomes pure JSON

Autoresearch wiring example:
    {
        "file_under_improvement": "supergraph/algos/graph.py",
        "metric_command": "python -m benchmarks.algos.bench_one graph --fast --json",
        "metric_format": "json",
        "metric_direction": "lower_is_better"
    }
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ALGO_TO_FILE = {
    "graph": "supergraph/algos/graph.py",
    "compact": "supergraph/algos/compact.py",
    "fusion": "supergraph/algos/fusion.py",
    "materialization": "supergraph/algos/materialization.py",
    "spreading": "supergraph/algos/spreading.py",
    "eviction": "supergraph/algos/eviction.py",
    "sort": "supergraph/algos/sort.py",
    "text": "supergraph/algos/text.py",
}

ALGO_TO_BENCH = {
    "graph": "benchmarks/algos/test_graph_bench.py",
    "compact": "benchmarks/algos/test_compact_bench.py",
    "fusion": "benchmarks/algos/test_fusion_bench.py",
    "materialization": "benchmarks/algos/test_materialization_bench.py",
    "spreading": "benchmarks/algos/test_spreading_bench.py",
    "eviction": "benchmarks/algos/test_eviction_bench.py",
    "sort": "benchmarks/algos/test_sort_bench.py",
    "text": "benchmarks/algos/test_text_bench.py",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _python() -> str:
    venv = _repo_root() / ".venv" / "bin" / "python"
    if venv.exists():
        return str(venv)
    return sys.executable


def run_bench(algo: str, fast: bool, quiet: bool) -> tuple[int, Path]:
    root = _repo_root()
    bench_path = root / ALGO_TO_BENCH[algo]
    json_dir = root / ".benchmarks"
    json_dir.mkdir(parents=True, exist_ok=True)
    json_out = json_dir / f"metric_{algo}.json"

    cmd = [
        _python(), "-m", "pytest",
        str(bench_path),
        "--benchmark-only",
        f"--benchmark-json={json_out}",
        "--no-header",
    ]
    if fast:
        cmd += [
            "--benchmark-max-time=0.3",
            "--benchmark-min-rounds=30",
            "--benchmark-warmup=off",
        ]
    if quiet:
        cmd += ["-q", "--tb=no"]

    result = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=quiet,
        text=True,
    )
    return result.returncode, json_out


def _unique_name(fullname: str, short_name: str) -> str:
    """Derive a stable unique metric key from pytest-benchmark fullname.

    fullname example:
        benchmarks/algos/test_graph_bench.py::TestBfsTraverse::test_1k_depth2
    → TestBfsTraverse::test_1k_depth2
    """
    if "::" in fullname:
        parts = fullname.split("::")
        return "::".join(parts[1:]) if len(parts) > 1 else fullname
    return short_name or fullname


def extract_metrics(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text())
    benches = data.get("benchmarks", [])
    metrics = []
    for b in benches:
        stats = b["stats"]
        fullname = b.get("fullname", "")
        short = b.get("name", "")
        unique = _unique_name(fullname, short)
        metrics.append({
            "name": unique,
            "fullname": fullname,
            "short_name": short,
            "group": b.get("group", ""),
            "mean_us": float(stats["mean"]) * 1_000_000.0,
            "min_us": float(stats["min"]) * 1_000_000.0,
            "max_us": float(stats["max"]) * 1_000_000.0,
            "stddev_us": float(stats["stddev"]) * 1_000_000.0,
            "median_us": float(stats.get("median", stats["mean"])) * 1_000_000.0,
            "rounds": int(stats.get("rounds", 0)),
        })
    return metrics


def write_summary(algo: str, metrics: list[dict]) -> Path:
    root = _repo_root()
    summary_path = root / ".benchmarks" / f"metric_{algo}_summary.json"
    summary = {
        "algo": algo,
        "file_under_improvement": ALGO_TO_FILE[algo],
        "unit": "microseconds",
        "lower_is_better": True,
        "stat": "min",
        "n_metrics": len(metrics),
        "metrics": {m["name"]: m["min_us"] for m in metrics},
        "detailed": metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-algo benchmark runner for autoresearch workflows",
    )
    parser.add_argument(
        "algo",
        choices=sorted(ALGO_TO_BENCH.keys()),
        help="which algo file to benchmark",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="short per-bench budget for tight iteration",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress pytest output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="replace METRIC lines with pure JSON on stdout",
    )
    args = parser.parse_args()

    rc, json_path = run_bench(args.algo, fast=args.fast, quiet=args.quiet or args.json)
    if rc != 0:
        print(f"ERROR: bench run failed for {args.algo} (exit {rc})",
              file=sys.stderr)
        return rc

    if not json_path.exists():
        print(f"ERROR: json output missing at {json_path}", file=sys.stderr)
        return 2

    try:
        metrics = extract_metrics(json_path)
    except Exception as e:
        print(f"ERROR: metric extraction failed: {e}", file=sys.stderr)
        return 3

    if not metrics:
        print(f"ERROR: no metrics collected for {args.algo}", file=sys.stderr)
        return 4

    summary_path = write_summary(args.algo, metrics)

    if args.json:
        payload = {
            "algo": args.algo,
            "file_under_improvement": ALGO_TO_FILE[args.algo],
            "unit": "microseconds",
            "lower_is_better": True,
            "stat": "min",
            "metrics": {m["name"]: m["min_us"] for m in metrics},
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0

    if not args.quiet:
        print()
        print(f"=== metric summary for {args.algo} ===")
        print(f"file_under_improvement:  {ALGO_TO_FILE[args.algo]}")
        print(f"unit:                    microseconds (lower = better)")
        print(f"n_metrics:               {len(metrics)}")
        print(f"summary_json:            {summary_path}")
        print()

    print(f"METRIC_FILE {ALGO_TO_FILE[args.algo]}")
    for m in metrics:
        print(f"METRIC {m['name']} {m['min_us']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
