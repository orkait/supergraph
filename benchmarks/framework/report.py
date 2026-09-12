
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .metrics import RunResult


def write_json(results: Iterable[RunResult], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in results]
    out.write_text(json.dumps(data, indent=2) + "\n")


def write_csv(results: Iterable[RunResult], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        d = r.to_dict()
        rows.append({
            "system": d["system"],
            "version": d["version"],
            "benchmark": d["benchmark"],
            "accuracy": d["quality"]["accuracy"],
            "recall_at_k": d["quality"]["recall_at_k"],
            "query_p50_ms": d["latency_query"]["p50_ms"],
            "query_p95_ms": d["latency_query"]["p95_ms"],
            "query_p99_ms": d["latency_query"]["p99_ms"],
            "ingest_mean_ms": d["latency_ingest"]["mean_ms"],
            "memory_peak_mb": d["memory"]["rss_peak_mb"],
            "total_tokens": d["cost"]["total_tokens"],
            "elapsed_s": d["total_elapsed_s"],
        })
    if not rows:
        return
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_markdown(results: Iterable[RunResult], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    results_list = list(results)

    lines: list[str] = []
    lines.append("# Agent Memory Benchmark Results")
    lines.append("")
    if results_list:
        lines.append(f"- Benchmark: **{results_list[0].benchmark}**")
        lines.append(f"- Systems:   **{len(results_list)}**")
        lines.append(f"- Run at:    {results_list[0].started_at}")
    lines.append("")

    lines.append("## Quality (higher is better)")
    lines.append("")
    lines.append("| System | Version | Accuracy | R@K | N |")
    lines.append("|---|---|---|---|---|")
    for r in sorted(results_list, key=lambda x: x.quality.accuracy, reverse=True):
        lines.append(
            f"| {r.system_name} | {r.system_version} "
            f"| {r.quality.accuracy:.3f} "
            f"| {r.quality.recall_at_k:.3f} "
            f"| {r.quality.n_questions} |"
        )
    lines.append("")

    lines.append("## Query latency, ms (lower is better)")
    lines.append("")
    lines.append("| System | p50 | p95 | p99 | mean | stddev |")
    lines.append("|---|---|---|---|---|---|")
    for r in results_list:
        lq = r.latency_query
        lines.append(
            f"| {r.system_name} "
            f"| {lq.p50:.1f} | {lq.p95:.1f} | {lq.p99:.1f} "
            f"| {lq.mean:.1f} | {lq.stddev:.1f} |"
        )
    lines.append("")

    lines.append("## Memory footprint")
    lines.append("")
    lines.append("| System | Peak RSS (MB) | Delta (MB) |")
    lines.append("|---|---|---|")
    for r in results_list:
        lines.append(
            f"| {r.system_name} "
            f"| {r.memory.rss_peak_mb:.1f} "
            f"| {r.memory.delta_mb:.1f} |"
        )
    lines.append("")

    lines.append("## Elapsed time + cost")
    lines.append("")
    lines.append("| System | Total elapsed (s) | Tokens |")
    lines.append("|---|---|---|")
    for r in results_list:
        lines.append(
            f"| {r.system_name} "
            f"| {r.total_elapsed_s:.1f} "
            f"| {r.cost.total_tokens} |"
        )
    lines.append("")

    has_categories = any(r.quality._categories for r in results_list)
    if has_categories:
        lines.append("## Quality by category")
        lines.append("")
        all_cats: set[str] = set()
        for r in results_list:
            all_cats.update(r.quality._categories.keys())
        sorted_cats = sorted(all_cats)
        header = "| Category | n | " + " | ".join(
            f"{r.system_name} acc / R@K" for r in results_list
        ) + " |"
        sep = "|---|---|" + "|".join(["---"] * len(results_list)) + "|"
        lines.append(header)
        lines.append(sep)
        for cat in sorted_cats:
            n = next(
                (r.quality._categories[cat].n for r in results_list
                 if cat in r.quality._categories),
                0,
            )
            cells = []
            for r in results_list:
                b = r.quality._categories.get(cat)
                if b:
                    cells.append(f"{b.accuracy:.3f} / {b.recall_at_k:.3f}")
                else:
                    cells.append("-")
            lines.append(f"| {cat} | {n} | " + " | ".join(cells) + " |")
        lines.append("")

    out.write_text("\n".join(lines) + "\n")
