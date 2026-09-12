---
title: First memory (LoCoMo walkthrough)
sidebar_position: 1
---

# First memory: LoCoMo walkthrough

End-to-end: populate a supergraph with a real conversation (LoCoMo conv-26: Caroline and Melanie, 19 sessions, 184 messages) and query it.

## Prerequisites

```bash
# Install supergraph (uv optional, pip works too)
uv sync

# Install Jina v5 Small embedder (1024d, 677M params)
SUPERGRAPH_MODEL_CACHE_DIR=/tmp/gs_models uv run python3 -c "
from supergraph.registry.installer import install_embedder, set_cache_dir
set_cache_dir('/tmp/gs_models')
install_embedder('jina-v5-small-retrieval')
"

# Download LoCoMo dataset
# Get locomo10.json from https://huggingface.co/datasets/Percena/locomo-mc10
# Place at /tmp/locomo/raw/locomo10.json
```

## Populate supergraph

```python
import os
os.environ['SUPERGRAPH_MODEL_CACHE_DIR'] = '/tmp/gs_models'

from benchmarks.framework.datasets import load_locomo
from benchmarks.framework.adapters.supergraph_ import SuperGraphAdapter

ds = load_locomo('/tmp/locomo', max_conversations=1)

config = {
    'embedder': 'installed',
    'embedder_model': 'jina-v5-small-retrieval',
    'embedder_cache_dir': '/tmp/gs_models',
    'embedder_gpu': True,
    'ceiling_mb': 512,
}
adapter = SuperGraphAdapter(config=config)
adapter.reset()

for sess in ds.records[0].sessions:
    adapter.ingest(sess)

gs = adapter._gs
print(f'Nodes: {gs.node_count}, Edges: {gs.edge_count}')
# Expected: 240 nodes (184 messages + 19 sessions + 37 entities), 784 edges
```

## Query examples

### REMEMBER (hybrid retrieval)

Vector + BM25 + graph + recency fused.

```python
# Single-hop factual
r = gs.execute('REMEMBER "What did Caroline research" LIMIT 10 WHERE kind = "message"')
for n in r.data[:3]:
    print(f'  score={n["_remember_score"]:.3f} {n.get("content","")[:70]}')
# Expected: "Caroline is researching adoption agencies" near top

# Multi-hop
r = gs.execute('REMEMBER "When did Caroline go to the LGBTQ support group" LIMIT 10 WHERE kind = "message"')

# Open-domain
r = gs.execute('REMEMBER "What pet does Caroline have" LIMIT 10 WHERE kind = "message"')

# With temporal anchor
r = gs.execute('REMEMBER "what happened" AT "2023-05-08" LIMIT 10 WHERE kind = "message"')
```

### RECALL (graph spreading activation)

```python
# From an entity - finds all messages connected via graph edges
r = gs.execute('RECALL FROM "ent:caroline" DEPTH 2 LIMIT 10')
for n in r.data[:5]:
    print(f'  [{n["kind"]}] {n.get("content", n.get("name", ""))[:50]}')

# From a specific message - finds neighbours
r = gs.execute('RECALL FROM "s1:msg0" DEPTH 2 LIMIT 10')
```

### SIMILAR TO (pure vector search)

```python
r = gs.execute('SIMILAR TO "counseling career psychology" LIMIT 5 WHERE kind = "message"')
for n in r.data:
    print(f'  sim={n["_similarity_score"]:.3f} {n.get("content","")[:60]}')
```

### LEXICAL SEARCH (BM25)

```python
r = gs.execute('LEXICAL SEARCH "adoption agency" LIMIT 5')
for n in r.data:
    print(f'  bm25={n["_bm25_score"]:.3f} {n.get("content","")[:60]}')
```

### TRAVERSE / SUBGRAPH (graph exploration)

```python
r = gs.execute('SUBGRAPH FROM "s1:msg0" DEPTH 2')
print(f'Nodes: {len(r.data["nodes"])}, Edges: {len(r.data["edges"])}')

ents = gs.execute('NODES WHERE kind = "entity"')
for e in ents.data[:10]:
    print(f'  {e["id"]}: {e.get("name","")}')
```

### Score breakdown

Every REMEMBER result includes per-signal scores:

```python
r = gs.execute('REMEMBER "Caroline counseling" LIMIT 1 WHERE kind = "message"')
n = r.data[0]
print(f'_remember_score: {n["_remember_score"]}')
print(f'_vector_sim:     {n["_vector_sim"]}')
print(f'_bm25_score:     {n["_bm25_score"]}')
print(f'_recency_score:  {n["_recency_score"]}')
print(f'_graph_score:    {n["_graph_score"]}')
print(f'_recall_score:   {n["_recall_score"]}')
```

## Retrieval recall (no LLM)

Tests direct LoCoMo evidence recall - strict (exact supporting message retrieved) and pragmatic (any message from a supporting session retrieved):

```bash
uv run python3 -m benchmarks.framework.ratchet_recall
```

## Full LoCoMo with LLM

1. Set API keys in `/.env` at the repo root (see `/.env.example`). The
   shared transport reads `OPENROUTER_API_KEY` + `OLLAMA_API_KEY`.
2. The preferred QA model is declared in
   `src/supergraph/llm_runner.py` as `QA_MODEL_PRIORITY`. Edit that list
   to swap models. Default: `gemma4:31b-cloud` (Ollama) with
   `google/gemma-4-31b-it` (OpenRouter) as fallback.

Run:

```bash
# 50Q random sample (~$0.07 on MiniMax nitro)
uv run python3 -c "
import os
os.environ['SUPERGRAPH_MODEL_CACHE_DIR'] = '/tmp/gs_models'
from benchmarks.framework.runners.locomo import run_locomo
from benchmarks.framework.datasets import load_locomo
from benchmarks.framework.adapters.supergraph_ import SuperGraphAdapter

ds = load_locomo('/tmp/locomo', max_conversations=1)
config = {
    'embedder': 'installed',
    'embedder_model': 'jina-v5-small-retrieval',
    'embedder_cache_dir': '/tmp/gs_models',
    'embedder_gpu': True,
    'ceiling_mb': 512,
}
adapter = SuperGraphAdapter(config=config)
summary, details = run_locomo(adapter, ds, k=10)
print(f'Overall F1: {summary[\"overall_f1\"]:.4f}')
"

# Full 1986Q (~$0.40 on MiniMax nitro)
uv run python3 -m benchmarks.framework.runners.locomo \
  --data-path /tmp/locomo \
  --embedder installed:jina-v5-small-retrieval \
  --k 10
```

## Graph structure after ingestion

```
240 nodes:
  184 messages  - conversation observations with [date] speaker: content
  19 sessions   - one per conversation session
  37 entities   - extracted via regex (Caroline, Melanie, LGBTQ, months, etc.)

784 edges:
  has_message   - session -> message
  mentions      - message -> entity
  next          - message -> next message (sequential order)
```

Each message has:

- Vector embedding (Jina v5 Small, 1024d)
- FTS5 index entry (BM25 searchable)
- `__event_at__` timestamp (from session date)
- Entity edges (mentions)
- Sequential edges (next)
