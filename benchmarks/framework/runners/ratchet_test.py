"""Ratchet test runner for LoCoMo retrieval improvements.

Runs 50Q (random 10/cat, seed=42) with configurable retrieval settings.
Usage: uv run python3 -m benchmarks.framework.runners.ratchet_test --label baseline
"""

from __future__ import annotations

import os
import random
import sys
import json
import time
from collections import defaultdict
from pathlib import Path

os.environ['SUPERGRAPH_MODEL_CACHE_DIR'] = '/tmp/gs_models'

from .locomo import run_locomo
from ..datasets import load_locomo
from ..adapters.supergraph_ import SuperGraphAdapter


def run_test(label: str, config: dict, k: int = 10, reranker=None) -> dict:
    ds = load_locomo('/tmp/locomo', max_conversations=1)

    random.seed(42)
    by_cat = defaultdict(list)
    for rec in ds.records:
        by_cat[rec.question.category].append(rec)

    sampled = []
    for cat, recs in sorted(by_cat.items()):
        chosen = random.sample(recs, min(10, len(recs)))
        sampled.extend(chosen)
    ds.records = sampled

    print(f'[{label}] {len(sampled)} Qs (random 10/cat, seed=42)', flush=True)

    adapter = SuperGraphAdapter(config=config)
    t0 = time.perf_counter()
    summary, details = run_locomo(adapter, ds, k=k, reranker=reranker)
    elapsed = time.perf_counter() - t0

    print(f'\n[{label}] Results ({elapsed:.1f}s):')
    for cat, v in summary['by_category'].items():
        print(f'  {cat:<20} n={v["n"]:<4} f1={v["f1"]:.4f}')
    print(f'  {"OVERALL":<20} n={summary["n_questions"]:<4} f1={summary["overall_f1"]:.4f}')
    print(flush=True)

    # Save
    out_dir = Path('benchmarks/framework/results')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'ratchet_{label}.json'
    out_path.write_text(json.dumps({'summary': summary, 'details': details, 'label': label}, indent=2))

    return summary


BASE_CONFIG = {
    'embedder': 'installed',
    'embedder_model': 'jina-v5-small-retrieval',
    'embedder_cache_dir': '/tmp/gs_models',
    'embedder_gpu': True,
    'ceiling_mb': 512,
    'search_oversample': 20,
    'retrieval_depth': 8,
    'recall_depth': 4,
    'max_query_entities': 1,
    'similar_to_oversample': 8,
    'lexical_search_oversample': 4,
    'retrieval_strategy': 'full',
}


if __name__ == '__main__':
    label = sys.argv[1] if len(sys.argv) > 1 else 'baseline'
    run_test(label, BASE_CONFIG)
