"""Benchmark CLI entry point.

Usage:
    python -m benchmarks.framework.cli list
    python -m benchmarks.framework.cli run --dataset longmemeval --data-path ./data --variant s
    python -m benchmarks.framework.cli run --dataset locomo --data-path ./data
    python -m benchmarks.framework.cli run --dataset beam --data-path /tmp/BEAM --chat-size 16k --end-index 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .adapters import AVAILABLE, get_adapter
from .datasets import DATASET_LOADERS
from .report import write_csv, write_json, write_markdown
from .runners.runner import run_benchmark

SUPPORTED_DATASETS = ["longmemeval", "locomo", "beam"]


def cmd_list(args: argparse.Namespace) -> int:
    print("Available adapters:")
    for name in sorted(AVAILABLE):
        cls = AVAILABLE[name]
        version = getattr(cls, "version", "unknown")
        print(f"  {name:<15} v{version}")
    print()
    print("Available benchmarks:")
    for name in SUPPORTED_DATASETS:
        print(f"  {name}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.dataset == "beam":
        return _run_beam(args)
    return _run_framework(args)


def _run_beam(args: argparse.Namespace) -> int:
    """Dispatch to BEAM benchmark runner (own protocol)."""
    from .runners.beam import main as beam_main

    beam_argv = [
        "--beam-root", args.data_path,
        "--chat-size", args.variant or "16k",
        "--start-index", str(getattr(args, "start_index", 1)),
        "--end-index", str(getattr(args, "end_index", 10)),
        "--k", str(args.k),
        "--ceiling-mb", str(args.ceiling_mb),
    ]
    if getattr(args, "embedder", None):
        beam_argv.extend(["--embedder", args.embedder])
    if getattr(args, "reader_model", None):
        beam_argv.extend(["--reader-model-name", args.reader_model])
    if getattr(args, "reader_url", None):
        beam_argv.extend(["--reader-model-url", args.reader_url])
    return beam_main(beam_argv)


def _run_framework(args: argparse.Namespace) -> int:
    """Dispatch to framework runner (longmemeval, locomo)."""
    adapter_cls = get_adapter(args.system)
    adapter = adapter_cls(
        config={
            "ceiling_mb": args.ceiling_mb,
            "queued": args.queued,
        }
    )

    if args.dataset not in DATASET_LOADERS:
        print(f"Unknown dataset: {args.dataset}", file=sys.stderr)
        return 2

    loader = DATASET_LOADERS[args.dataset]
    if args.dataset == "longmemeval":
        dataset = loader(args.data_path, variant=args.variant)
    else:
        dataset = loader(args.data_path)

    result = run_benchmark(
        adapter,
        dataset,
        k=args.k,
        max_questions=args.max_questions,
        config={
            "ceiling_mb": args.ceiling_mb,
            "queued": args.queued,
            "k": args.k,
            "variant": args.variant,
            "max_questions": args.max_questions,
        },
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.started_at.replace(":", "-")[:19]
    prefix = out_dir / f"{args.system}_{args.dataset}_{args.variant}_{stamp}"
    write_json([result], prefix.with_suffix(".json"))
    write_csv([result], prefix.with_suffix(".csv"))
    write_markdown([result], prefix.with_suffix(".md"))

    print()
    print(f"Results: {prefix}.{{json,csv,md}}")
    print(f"  accuracy    {result.quality.accuracy:.3f}")
    print(f"  recall@{args.k}    {result.quality.recall_at_k:.3f}")
    print(
        f"  latency     p50={result.latency_query.p50:.1f}ms "
        f"p95={result.latency_query.p95:.1f}ms "
        f"p99={result.latency_query.p99:.1f}ms"
    )
    print(
        f"  memory      peak={result.memory.rss_peak_mb:.1f}MB "
        f"delta={result.memory.delta_mb:.1f}MB"
    )
    print(f"  elapsed     {result.total_elapsed_s:.1f}s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="benchmarks.framework.cli",
        description="SuperGraph benchmark runner (LongMemEval, LoCoMo, BEAM)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List available benchmarks")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="Run a benchmark")
    p_run.add_argument("--system", default="supergraph", choices=sorted(AVAILABLE.keys()))
    p_run.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    p_run.add_argument("--data-path", required=True, type=str)
    p_run.add_argument("--variant", default="s", help="LongMemEval: s/m/l. BEAM: chat size (16k/64k/256k)")
    p_run.add_argument("--k", type=int, default=5)
    p_run.add_argument("--max-questions", type=int, default=None)
    p_run.add_argument("--ceiling-mb", type=int, default=2048)
    p_run.add_argument("--queued", action="store_true")
    p_run.add_argument("--out-dir", default="benchmarks/framework/results")
    # BEAM-specific
    p_run.add_argument("--start-index", type=int, default=1)
    p_run.add_argument("--end-index", type=int, default=10)
    p_run.add_argument("--embedder", default=None)
    p_run.add_argument("--reader-model", default=None, help="BEAM: LLM model for answer generation")
    p_run.add_argument("--reader-url", default=None, help="BEAM: OpenAI-compatible base URL")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
