---
title: Architecture
sidebar_position: 1
---

# Architecture

Three storage engines, one typed DSL, a tiered ingest pipeline, and a hybrid retrieval engine that fuses all of them.

<p align="center">
  <img src="/img/architecture.svg" alt="supergraph architecture: DSL + three storage engines + ingest pipeline + retrieval" width="780" />
</p>

## What flows where

- The **DSL** (Lark LALR(1), ~70 verbs) is the only way in. Every `CREATE`, `UPDATE`, `DELETE`, `ASSERT`, `RETRACT`, `INGEST`, `SYS *` goes through it.
- **Direct writes** (`CREATE NODE`, `ASSERT`, `UPDATE`, `CREATE EDGE`, ...) land straight in the three engines:

### Three engines

- **Graph** - typed numpy columns + scipy CSR edges. Reserved columns like `__event_at__`, `__confidence__`, `__retracted__` live here. See the [edge matrix internals](./edge-matrix).
- **Vector** - usearch HNSW with cosine. Auto-populated via schema `EMBED content` or the `DOCUMENT "..."` clause.
- **Document** - SQLite + FTS5 virtual table. BM25 + blob storage + single-owner path lock.

### Ingest pipeline

`INGEST "file.ext"` is itself a DSL verb. It dispatches to the ingest pipeline, which is tiered and modality-aware:

| Format | Parser |
|---|---|
| `txt`, `md` | direct |
| `html`, `docx`, `xlsx` | markitdown |
| `pdf` | pymupdf4llm, docling fallback |
| `png`, `jpg` | vision sidecar (local llama.cpp + SmolVLM2-2.2B default, `[vision]` extra) |
| `wav`, `mp3`, `flac`, `m4a` | whisper in-process (faster-whisper, `[audio]` extra) |

Pipeline output flows into the same three engines.

### Retrieval

`REMEMBER` / `RECALL` / `SIMILAR TO` / `LEXICAL SEARCH` / `TRAVERSE` read from all three engines and fuse the signals. See [REMEMBER pipeline](./remember-pipeline) for the fusion internals.

## Single-owner per path

Persistent stores take an advisory lock on `<path>/.supergraph.lock`. A second `SuperGraph(path=...)` against the same path raises `StoreInUse` - WAL replay + compact + checkpoint are not safe across processes. In-memory stores are unlocked. OS reclaims the lock on process exit.

## Thread safety

Default is single-threaded. `queued=True` installs a worker thread that serialises writes from multiple callers. BLAS thread count is capped at import time (`OMP_NUM_THREADS=2` by default; override with `SUPERGRAPH_BLAS_CAP=N`) so importing supergraph doesn't saturate all cores.
