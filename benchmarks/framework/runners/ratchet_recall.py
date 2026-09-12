
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("SUPERGRAPH_MODEL_CACHE_DIR", "/tmp/gs_models")
logging.getLogger("supergraph.events").setLevel(logging.WARNING)

from ..datasets import load_locomo
from ..adapters.supergraph_ import SuperGraphAdapter


def build_evidence_lookup(conv: dict) -> dict[str, str]:
    lookup: dict[str, str] = {}
    observations = conv.get("observation", {})
    sess_idx = 1
    while f"session_{sess_idx}_observation" in observations:
        obs_key = f"session_{sess_idx}_observation"
        msg_idx = 0
        for _speaker, facts in observations[obs_key].items():
            for _fact_text, evidence_id in facts:
                msg_id = f"s{sess_idx}:msg{msg_idx}"
                if isinstance(evidence_id, list):
                    for eid in evidence_id:
                        lookup[eid] = msg_id
                else:
                    lookup[evidence_id] = msg_id
                msg_idx += 1
        sess_idx += 1
    return lookup


def build_evidence_lookups(raw_conversations: list[dict]) -> dict[str, dict[str, str]]:
    return {
        conv.get("sample_id", f"conv-{idx}"): build_evidence_lookup(conv)
        for idx, conv in enumerate(raw_conversations)
    }


def score_evidence_support(
    *,
    evidence_ids: list[str],
    evidence_lookup: dict[str, str],
    retrieved_ids: list[str],
) -> dict[str, float | bool]:
    expected_ids = [evidence_lookup[eid] for eid in evidence_ids if eid in evidence_lookup]
    expected_set = set(expected_ids)
    retrieved_set = set(retrieved_ids)

    strict_hits = expected_set & retrieved_set
    strict_total = len(expected_set)
    strict_hit = len(strict_hits) > 0
    strict_coverage = len(strict_hits) / strict_total if strict_total else 0.0

    expected_sessions = {msg_id.split(":")[0] for msg_id in expected_set}
    retrieved_sessions = {msg_id.split(":")[0] for msg_id in retrieved_set}
    pragmatic_hits = expected_sessions & retrieved_sessions
    pragmatic_total = len(expected_sessions)
    pragmatic_hit = len(pragmatic_hits) > 0
    pragmatic_coverage = (
        len(pragmatic_hits) / pragmatic_total if pragmatic_total else 0.0
    )

    return {
        "strict_hit": strict_hit,
        "strict_coverage": strict_coverage,
        "pragmatic_hit": pragmatic_hit,
        "pragmatic_coverage": pragmatic_coverage,
    }


def run(label: str = "test", k: int = 10, max_conversations: int | None = None) -> dict:
    raw = json.loads(Path("/tmp/locomo/raw/locomo10.json").read_text())
    if max_conversations is not None:
        raw = raw[:max_conversations]
    ds = load_locomo("/tmp/locomo", max_conversations=max_conversations)
    evidence_lookups = build_evidence_lookups(raw)

    config = {
        "embedder": "installed",
        "embedder_model": "jina-v5-small-retrieval",
        "embedder_cache_dir": "./models",
        "embedder_gpu": False,
        "ceiling_mb": 512,
    }

    totals = {
        "strict_hit": 0,
        "strict_coverage_sum": 0.0,
        "pragmatic_hit": 0,
        "pragmatic_coverage_sum": 0.0,
        "n": 0,
    }
    by_cat: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "strict_hit": 0,
            "strict_coverage_sum": 0.0,
            "pragmatic_hit": 0,
            "pragmatic_coverage_sum": 0.0,
            "n": 0,
        }
    )

    records_by_conv: dict[str, list] = defaultdict(list)
    sessions_by_conv: dict[str, list] = {}
    for rec in ds.records:
        conv_id = rec.question.metadata.get("sample_id", "unknown")
        records_by_conv[conv_id].append(rec)
        if conv_id not in sessions_by_conv:
            sessions_by_conv[conv_id] = rec.sessions

    adapter = SuperGraphAdapter(config=config)
    try:
        for conv_id, records in records_by_conv.items():
            evidence_lookup = evidence_lookups.get(conv_id, {})
            adapter.reset()
            for sess in sessions_by_conv[conv_id]:
                adapter.ingest(sess)
            gs = adapter._gs

            for rec in records:
                evidence_ids = rec.question.metadata.get("evidence") or []
                if not evidence_ids:
                    continue
                q = rec.question.question.replace('"', '\\"')
                result = gs.execute(f'REMEMBER "{q}" LIMIT {k} WHERE kind = "message"')
                retrieved_ids = [node["id"] for node in result.data]
                scores = score_evidence_support(
                    evidence_ids=evidence_ids,
                    evidence_lookup=evidence_lookup,
                    retrieved_ids=retrieved_ids,
                )
                cat = rec.question.category or "unknown"
                totals["n"] += 1
                totals["strict_hit"] += 1 if scores["strict_hit"] else 0
                totals["strict_coverage_sum"] += float(scores["strict_coverage"])
                totals["pragmatic_hit"] += 1 if scores["pragmatic_hit"] else 0
                totals["pragmatic_coverage_sum"] += float(scores["pragmatic_coverage"])

                bucket = by_cat[cat]
                bucket["n"] += 1
                bucket["strict_hit"] += 1 if scores["strict_hit"] else 0
                bucket["strict_coverage_sum"] += float(scores["strict_coverage"])
                bucket["pragmatic_hit"] += 1 if scores["pragmatic_hit"] else 0
                bucket["pragmatic_coverage_sum"] += float(scores["pragmatic_coverage"])
    finally:
        adapter.close()

    n = max(int(totals["n"]), 1)
    summary = {
        "label": label,
        "n": int(totals["n"]),
        "strict_hit_rate": totals["strict_hit"] / n,
        "strict_coverage": totals["strict_coverage_sum"] / n,
        "pragmatic_hit_rate": totals["pragmatic_hit"] / n,
        "pragmatic_coverage": totals["pragmatic_coverage_sum"] / n,
        "by_category": {},
    }

    for cat in sorted(by_cat):
        bucket = by_cat[cat]
        n_cat = max(int(bucket["n"]), 1)
        summary["by_category"][cat] = {
            "n": int(bucket["n"]),
            "strict_hit_rate": bucket["strict_hit"] / n_cat,
            "strict_coverage": bucket["strict_coverage_sum"] / n_cat,
            "pragmatic_hit_rate": bucket["pragmatic_hit"] / n_cat,
            "pragmatic_coverage": bucket["pragmatic_coverage_sum"] / n_cat,
        }

    print(f"{label}: n={summary['n']}")
    print(
        "  strict    "
        f"hit={summary['strict_hit_rate']:.3f} "
        f"coverage={summary['strict_coverage']:.3f}"
    )
    print(
        "  pragmatic "
        f"hit={summary['pragmatic_hit_rate']:.3f} "
        f"coverage={summary['pragmatic_coverage']:.3f}"
    )
    for cat, vals in summary["by_category"].items():
        print(
            f"  {cat:<20} "
            f"strict={vals['strict_hit_rate']:.3f} "
            f"prag={vals['pragmatic_hit_rate']:.3f} "
            f"n={vals['n']}"
        )

    return summary


if __name__ == "__main__":
    import sys

    label = sys.argv[1] if len(sys.argv) > 1 else "test"
    run(label)
