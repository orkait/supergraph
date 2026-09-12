---
title: DSL reference
sidebar_position: 1
---

# DSL reference

Every verb in the supergraph DSL, grouped by role. For the typed Python API, see [Query builder](../query-builder).

## Reads

```sql
NODE "id"
NODE "id" WITH DOCUMENT
NODES WHERE kind = "memory" AND importance > 0.5 LIMIT 10
EDGES FROM "id" WHERE kind = "calls"
TRAVERSE FROM "id" DEPTH 3
SUBGRAPH FROM "id" DEPTH 2
PATH FROM "a" TO "b" MAX_DEPTH 5
SHORTEST PATH FROM "a" TO "b"
ANCESTORS OF "id" DEPTH 3
DESCENDANTS OF "id" DEPTH 3
COMMON NEIGHBORS OF "a" AND "b"
MATCH ("fn_main") -[kind = "calls"]-> (callee)
COUNT NODES WHERE kind = "memory"
AGGREGATE NODES GROUP BY kind SELECT COUNT()
RECALL FROM "id" DEPTH 3 LIMIT 10
SIMILAR TO "text" LIMIT 10
SIMILAR TO NODE "id" LIMIT 10
SIMILAR TO [0.1, 0.2, ...] LIMIT 10
LEXICAL SEARCH "phrase" LIMIT 10
REMEMBER "query" LIMIT 10
REMEMBER "query" AT "2024-03" TOKENS 4000
ANSWER "question" LIMIT 5
ANSWER "question" LIMIT 5 USING "reader-name"
WHAT IF RETRACT "id"
```

## Writes

```sql
CREATE NODE "id" kind = "x" name = "foo"
CREATE NODE "id" kind = "x" EVENT_AT "2024-03-15"
CREATE NODE "id" kind = "x" EXPIRES IN 1h DOCUMENT "full text..."
UPDATE NODE "id" SET name = "new"
UPSERT NODE "id" kind = "x" name = "foo"
DELETE NODE "id"
DELETE NODES WHERE kind = "test"
UPDATE NODES WHERE kind = "fact" SET confidence = 0.5
CREATE EDGE "src" -> "tgt" kind = "calls"
INCREMENT NODE "id" hits BY 1
ASSERT "id" kind = "fact" value = 42 CONFIDENCE 0.9 SOURCE "tool" EVENT_AT "2024-01"
RETRACT "id" REASON "outdated"
MERGE NODE "old" INTO "canonical"
PROPAGATE "id" FIELD confidence DEPTH 3
INGEST "file.pdf" AS "doc:q3" KIND "report"
FORGET NODE "id"
BIND CONTEXT "session-1"
DISCARD CONTEXT "session-1"
BEGIN ... COMMIT
```

`DOCUMENT` auto-populates BM25 + vector + blob in one shot. `EXPIRES IN` must come before `DOCUMENT` in the clause order.

## System

```sql
SYS STATUS / SYS STATS / SYS HEALTH
SYS KINDS / SYS EDGE KINDS / SYS DESCRIBE NODE "memory"
SYS REGISTER NODE KIND "memory" REQUIRED topic:string EMBED content
SYS CONNECT / SYS CONNECT THRESHOLD 0.9
SYS CONSOLIDATE THRESHOLD 0.7
SYS DUPLICATES THRESHOLD 0.95
SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic
SYS EXPIRE WHERE kind = "working"
SYS SNAPSHOT "name" / SYS ROLLBACK TO "name"
SYS EMBEDDERS / SYS REEMBED
SYS RETAIN / SYS EVICT
SYS CHECKPOINT / SYS REBUILD INDICES / SYS CLEAR CACHE
SYS OPTIMIZE / SYS OPTIMIZE COMPACT
SYS LOG LIMIT 20 / SYS LOG TRACE "id"
SYS CRON ADD "name" SCHEDULE "0 * * * *" QUERY "SYS EXPIRE"
SYS EVOLVE RULE "name" WHEN signal OP value THEN action COOLDOWN n
SYS EVOLVE LIST / SHOW / ENABLE / DISABLE / DELETE / HISTORY
```

## Common patterns

### Store a retrievable memory

```sql
CREATE NODE "mem:123" kind = "memory" topic = "finance"
  DOCUMENT "Q3 revenue beat expectations driven by enterprise renewals."
```

`DOCUMENT` populates vector + BM25 + blob in one shot. Without it, the node stores typed columns only and `REMEMBER` / `LEXICAL` return zero.

### Hybrid retrieval

```sql
REMEMBER "quarterly revenue trends" TOKENS 4000
```

5-signal fusion over vector + BM25 + recency + graph + confidence. See [REMEMBER pipeline](../concepts/remember-pipeline).

Every result carries per-signal scores on every node (`_remember_score`, `_vector_sim`, `_bm25_score`, `_recency_score`, `_graph_score`, `_co_bonus`, `_recall_boost`, `_rank_stage`) and a rich `Result.meta["signals"]` block describing fusion weights, recency half-life, per-stage candidate counts, reranker state, and nucleus state.

### Dry-run the pipeline

```sql
SYS EXPLAIN REMEMBER "quarterly revenue trends" LIMIT 3
```

Returns the candidate plan with per-signal scores without materializing nodes, running the reranker, triggering nucleus, or mutating recall counts. Safe for iterative tuning.

### Retrieval-augmented answer

```sql
ANSWER "What were the biggest Q3 revenue drivers?" LIMIT 5
ANSWER "Who attended the LGBTQ support group?" LIMIT 5 USING "careful-reader"
```

Runs REMEMBER internally, hands retrieved passages + question to a reader LLM configured at SuperGraph construction (`reader=` or `readers={...}`), returns `{answer, cited_slots, candidates, reader}` plus the same `meta["signals"]` telemetry. SuperGraph ships no LLM dependency; the reader is a user-supplied callable.

### Temporal retrieval

```sql
REMEMBER "what happened in May" AT "2024-05" LIMIT 10
```

Recency-weighted + hard filter on AT window.

### Beliefs

```sql
ASSERT "fact:earth-radius" value = 6371 kind = "fact" CONFIDENCE 0.99 SOURCE "physics-tool"
RETRACT "fact:old-preference" REASON "user corrected this"
SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic
```

### Temporal anchoring

```sql
CREATE NODE "event:trip" kind = "event" content = "visited Paris" EVENT_AT "2024-03-15"
REMEMBER "trip plans" AT "2024-03" LIMIT 10
```

### Consolidation

```sql
SYS CONSOLIDATE THRESHOLD 0.7
```

Cluster episodic memories, no LLM needed.

### TTL + hard delete

```sql
CREATE NODE "scratch:temp" kind = "working" data = "..." EXPIRES IN 30m
SYS EXPIRE WHERE kind = "working"
FORGET NODE "mem:old"
```

### Snapshots

```sql
SYS SNAPSHOT "before-hypothesis"
SYS ROLLBACK TO "before-hypothesis"
```

### Scheduled maintenance

Requires `queued=True`.

```sql
SYS CRON ADD "expire-ttl" SCHEDULE "@hourly" QUERY "SYS EXPIRE"
```

### Self-tuning rules

```sql
SYS EVOLVE RULE "reindex-on-drift"
  WHEN recall_hit_rate <= 0.4
  THEN RUN SYS REEMBED
  COOLDOWN 86400
```

### Markdown vault

```python
# python: SuperGraph(path="./brain", vault="./notes")
```

```sql
VAULT NEW "Project Requirements" KIND "context"
VAULT SEARCH "deployment requirements" LIMIT 5
```

### Context isolation

```sql
BIND CONTEXT "reasoning-session-42"
CREATE NODE "hyp:1" kind = "hypothesis" content = "maybe X"
DISCARD CONTEXT "reasoning-session-42"
```

## Grammar

Lark LALR(1), source of truth at [`src/supergraph/dsl/grammar.lark`](https://github.com/orkait/supergraph/blob/main/src/supergraph/dsl/grammar.lark).
