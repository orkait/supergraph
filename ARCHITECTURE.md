# Architecture

supergraph has two parts. Nothing else.

```
┌─────────────────────────────────────────────────────────────┐
│ superclaw/                           the harness            │
│   identity · permission · modes · planning · compaction     │
│   delegation · completion                                   │
└───────────────────────────┬─────────────────────────────────┘
                            │ DSL over HTTP, MCP, or in-process
┌───────────────────────────▼─────────────────────────────────┐
│ src/supergraph/                      the substrate          │
│                                                             │
│   dsl/         grammar.lark → AST → handler registry        │
│   core/        slot arrays · ColumnStore · CSR edges        │
│   query/       typed Python builder over the same grammar   │
│   persistence/ sqlite blobs + WAL                           │
│   vector/      usearch HNSW                                 │
│   document/    sqlite blobs + FTS5 BM25                     │
│   embedding/   ONNX / GGUF / model2vec embedders, rerankers │
│   ingest/      files, media, NL→DSL                         │
│   algos/       pure numpy/scipy, no supergraph imports      │
└─────────────────────────────────────────────────────────────┘
```

## The substrate

An in-memory columnar graph, checkpointed to sqlite, with a hand-written LALR
DSL as its only interface.

| Layer | What |
|---|---|
| Storage | slot-indexed numpy arrays, typed columns with presence bitmaps, interned strings, scipy CSR edges with an LSM-style dynamic-edge buffer |
| Persistence | four stores, not one transaction: sqlite blobs, `vectors.usearch`, `documents.db`, and the `wal` table |
| Retrieval | `REMEMBER` fuses vector, BM25, recency and graph signals, then reranks, then optionally expands the nucleus |
| Beliefs | `ASSERT` / `RETRACT` with `__confidence__`, `__source__`, soft retraction that preserves history |
| Time | `__created_at__`, `__updated_at__`, `__event_at__`, `__expires_at__` with `SYS EXPIRE` |
| Metacognition | agent-written `WHEN/THEN` rules that retune the engine during health ticks |

Single-writer by design. `queued=True` installs a submission queue so callers
can share an instance across threads; it is not concurrent execution. A
cross-process `flock` on `.supergraph.lock` keeps two processes off one path.

## The harness

See `superclaw/README.md`. A terminal coding agent: one loop, a tool registry
behind a permission gate, compaction and guardrails, with sessions, plan state
and long-term memory stored in the substrate instead of on disk. Runs as the
`superclaw` command.

What the harness writes into the substrate, all through `gs.execute`:

| Node kind | Meaning | Namespace | Lifetime |
|---|---|---|---|
| `session`, `event` | the session record: prompt, message, tool_result, plan, error | `superclaw` | kept |
| `obs` | tool output bodies behind a `§ref` | `superclaw` | kept; web pages `EXPIRES IN 7d` |
| `memory` | facts the user stated | default | kept, optional `EXPIRES` |
| `fact` | facts learned from a source, asserted with confidence, source and event time | `superclaw` | kept until `RETRACT` |
| `kernel` | python namespace checkpoint per session | `superclaw` | overwritten |
| `cronjob` | scheduled prompts | `superclaw` | until deleted |

Edges: `session -> session` (`fork`), `session -> file` (`read`, `wrote`), `obs -> session` (`produced`), `fact -> session` (`learned_in`), `fact -> obs` (`from`), `fact -> fact` (`supersedes`). `file` nodes hold the absolute path and live in the `superclaw` namespace. Memories carry no edges: they sit in the default namespace, and an edge across namespaces is accepted but invisible from both sides.

## Why they live together

The prompt-layer harnesses in the field cannot date a stored fact or expire
one. Every one of them ships a staleness warning and none ships a mechanism.
The substrate already has valid time, transaction time, TTL with automatic
expiry, soft retraction, contradiction detection and additive
episodic-to-semantic consolidation.

That is the whole reason the harness sits next to the substrate rather than on
top of a vector store.

## Going deeper

This file is the two-part map. For the substrate internals - the three storage
engines, the DSL verb surface, the ingest tiers and the retrieval fusion - see
[website/docs/concepts/architecture.md](website/docs/concepts/architecture.md).

## Design docs

| Doc | Status |
|---|---|
| `design/tiered-storage.md` | proposed, unimplemented. Hot numpy tier plus a cold sqlite tier, with `SYS ARCHIVE` / `SYS PROMOTE` |
| `design/compaction-sentinel.md` | proposed, unimplemented. Makes compaction crash-recoverable, and closes the dirty-flag gap that lets `SYS EXPIRE` and `SYS EVICT` lose their work on reopen |
