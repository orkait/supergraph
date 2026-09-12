---
title: REMEMBER pipeline
sidebar_position: 2
---

# REMEMBER pipeline

`REMEMBER` is the core retrieval command. Five-stage pipeline, four weighted signals, optional rerank and nucleus walk.

<p align="center">
  <img src="/img/remember.svg" alt="REMEMBER 5-stage retrieval pipeline: gather to fuse to temporal to rerank to nucleus" width="620" />
</p>

## Signals fused at stage 2

Defaults; weights configurable.

| Signal | Weight | Source |
|---|---|---|
| `vec_signal` | 0.52 | max sentence cosine over usearch ANN |
| `bm25_signal` | 0.25 | SQLite FTS5 over `doc_fts` |
| `recency` | 0.15 | `exp(-age / half_life)` from `__event_at__` or `__updated_at__` |
| `graph_signal` | 0.08 | entity-degree sum over mentioned entities (opt-in) |
| + co-occurrence | bonus | `min(vec, bm25) * 0.10` when a candidate is found by both |
| + recall-frequency | nudge | `log1p(recall_count) * 0.05` |

## Configuration

All weights and knobs are configurable via `supergraph.json`, `SUPERGRAPH_DSL_*` env vars, or constructor kwargs. See [Configuration](../configuration).

## Score breakdown

Every `REMEMBER` result includes per-signal scores:

```python
r = g.execute('REMEMBER "Caroline counseling" LIMIT 1 WHERE kind = "message"')
n = r.data[0]
print(f'_remember_score: {n["_remember_score"]}')
print(f'_vector_sim:     {n["_vector_sim"]}')
print(f'_bm25_score:     {n["_bm25_score"]}')
print(f'_recency_score:  {n["_recency_score"]}')
print(f'_graph_score:    {n["_graph_score"]}')
print(f'_recall_score:   {n["_recall_score"]}')
```

Good for debugging why a candidate ranked where it did.
