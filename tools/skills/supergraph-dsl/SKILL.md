---
name: supergraph-dsl
description: Reference for supergraph's DSL (Lark LALR(1) grammar, ~70 verbs). Every verb, every clause, every escape rule, every retrieval gotcha. Use when emitting supergraph DSL statements at runtime as a string - LLM-driven ingestor / distiller / agent memory writer / query generator. Covers reads (REMEMBER, RECALL, SIMILAR, LEXICAL, TRAVERSE, MATCH), writes (CREATE NODE, ASSERT, RETRACT, INGEST), SYS ops, VAULT ops, BEGIN/COMMIT batches, WHERE expressions, pattern matching, and string escape rules. For the typed Python builder (`q`/`F`/`P`), use `supergraph-builder` skill instead.
compatibility: supergraph >= 0.3.0.
metadata:
  author: orkait
  version: "1.0"
---

# supergraph DSL

Emit supergraph DSL as text. One statement per line. String-escape `"` → `\"` inside literals. Parser rejects invalid grammar - adapter drops bad lines and continues.

## When

- LLM ingestor converting raw turns → stored memories
- LLM distiller emitting `ASSERT` / `RETRACT` from new evidence
- Query generator emitting DSL in response to natural-language asks
- Any runtime that produces DSL strings for `gs.execute(...)`

For Python adapter code (imports supergraph as a library and uses `q.create_node(...)`), see `supergraph-builder` skill.

## 3 engines behind 1 DSL

| Engine | Stores | Populated by |
|---|---|---|
| Graph (numpy columns + scipy CSR edges) | typed fields, `__event_at__`, `__confidence__`, `__retracted__`, `__source__` | CREATE NODE + CREATE EDGE + ASSERT |
| Vector (usearch HNSW, cosine) | (slot, embedding) | `EMBED content` schema OR `DOCUMENT "text"` clause |
| Document (SQLite + FTS5 `doc_fts` + blobs) | BM25 index + blob | `DOCUMENT "text"` clause OR `INGEST` |

One node can live in all three.

## Primitive → layer map

| Primitive | Hits | Use when |
|---|---|---|
| `NODE "id"` | columns | Know the id |
| `NODES WHERE ...` | columns | Structured filter |
| `SIMILAR TO "text"` | vectors | NL cue, no anchor |
| `SIMILAR TO NODE "id"` | vectors | "Like this one" |
| `LEXICAL SEARCH "text"` | FTS5 | Distinctive keywords |
| `REMEMBER "text"` | all three + graph signal | Default NL retrieval |
| `RECALL FROM "id" DEPTH k` | edges (spreading) | Anchor → connected |
| `TRAVERSE` / `PATH` / `ANCESTORS` / `MATCH` | edges (deterministic) | Structured walks |

## Ingestion order (critical)

1. `SYS REGISTER NODE KIND "kind" REQUIRED f:t,... [OPTIONAL ...] [EMBED field]` - first, before any CREATE. Pre-allocates numpy dtype. `EMBED` marks auto-embed field.
2. `SYS REGISTER EDGE KIND "kind" FROM kind1,... TO kind1,...` - validates CREATE EDGE endpoints.
3. Bulk CREATEs inside `BEGIN...COMMIT` only for small atomic groups (parent + N children). For bulk loads do NOT use BEGIN/COMMIT - it snapshots all columns for rollback (O(cols × nodes)). Emit statements unwrapped instead.
4. Use `CREATE NODE "id" ... DOCUMENT "text"` - populates graph row + vector + FTS5 + blob in one statement.
5. Chain messages with `next` edges for RECALL/ANCESTORS walks.
6. Build entity graph: for each extracted entity, `UPSERT NODE "ent:<slug>" kind="entity" name="Name"` then `CREATE EDGE "msg:id" -> "ent:slug" kind="mentions"`. **Dedupe per-message** - NER may emit same entity twice from multi-span matches; duplicate edges crash BEGIN/COMMIT with BatchRollback.
7. Bulk-create edges before first read. CREATE EDGE sets dirty flag; first CSR-needing read triggers O(total_edges) rebuild.
8. Do NOT EMBED entity nodes - short names make noisy vectors. Entities only live as graph anchors.

## DSL reference

### Reads

```
NODE "id" [WITH DOCUMENT]
NODES [WHERE expr] [ORDER BY field [ASC|DESC]] [LIMIT n] [OFFSET n]
EDGES FROM|TO "id" [WHERE expr] [LIMIT n]
TRAVERSE FROM "id" DEPTH n [WHERE expr] [LIMIT n]
SUBGRAPH FROM "id" DEPTH n
PATH FROM "a" TO "b" MAX_DEPTH n [WHERE expr]
PATHS FROM "a" TO "b" MAX_DEPTH n [WHERE expr]
SHORTEST PATH FROM "a" TO "b" [MAX_DEPTH n] [WHERE expr]
WEIGHTED SHORTEST PATH FROM "a" TO "b" [MAX_DEPTH n] [WHERE expr]
DISTANCE FROM "a" TO "b" MAX_DEPTH n
WEIGHTED DISTANCE FROM "a" TO "b" [MAX_DEPTH n]
ANCESTORS OF "id" DEPTH n [WHERE expr]
DESCENDANTS OF "id" DEPTH n [WHERE expr]
COMMON NEIGHBORS OF "a" AND "b" [WHERE expr]
MATCH <pattern> [LIMIT n]
COUNT NODES|EDGES [WHERE expr]
AGGREGATE NODES [WHERE expr] [GROUP BY f,...] SELECT agg,... [HAVING agg OP v] [ORDER BY agg [ASC|DESC]] [LIMIT n]
RECALL FROM "id" DEPTH n [LIMIT n] [WHERE expr]
SIMILAR TO "text"|NODE "id"|[0.1,0.2,...] [LIMIT n] [WHERE expr]
LEXICAL SEARCH "text" [LIMIT n] [WHERE expr]
REMEMBER "text" [AT "2024-05"] [TOKENS n] [LIMIT n] [WHERE expr]
ANSWER "text" [AT "2024-05"] [TOKENS n] [LIMIT n] [WHERE expr] [USING "reader"]
WHAT IF RETRACT "id"
```

Agg funcs: `COUNT()`, `COUNT DISTINCT(f)`, `SUM(f)`, `AVG(f)`, `MIN(f)`, `MAX(f)`.

### Writes

```
CREATE NODE "id" f=v,... [VECTOR [0.1,...]] [EXPIRES IN n<smhd>|EXPIRES AT "date"] [EVENT_AT "date"] [DOCUMENT "text"]
CREATE NODE AUTO f=v,... [VECTOR ...] [EXPIRES ...] [EVENT_AT ...] [DOCUMENT ...]
UPDATE NODE "id" SET f=v,...
UPSERT NODE "id" f=v,... [VECTOR ...] [EXPIRES ...] [EVENT_AT ...]
DELETE NODE "id"
DELETE NODES WHERE expr                        -- WHERE required
UPDATE NODES WHERE expr SET f=v,...
CREATE EDGE "src"|$var -> "tgt"|$var f=v,...
UPDATE EDGE "src" -> "tgt" SET f=v,... [WHERE expr]
DELETE EDGE "src" -> "tgt" [WHERE expr]
DELETE EDGES FROM|TO "id" [WHERE expr]
INCREMENT NODE "id" field BY n                 -- n may be negative
ASSERT "id" f=v,... [CONFIDENCE n] [SOURCE "s"] [EVENT_AT "t"]   -- sets __confidence__
RETRACT "id" [REASON "r"]                      -- sets __retracted__=true
MERGE NODE "old" INTO "canonical"
PROPAGATE "id" FIELD f DEPTH n
CONNECT NODE "id" [THRESHOLD n]
FORGET NODE "id"                               -- hard delete
BIND CONTEXT "name"
DISCARD CONTEXT "name"
INGEST "file.ext" [AS "id"] [KIND "k"] [USING parser | USING VISION "model"]
```

**Clause order in CREATE NODE / UPSERT NODE** (required): `field_pairs`, `VECTOR`, `EXPIRES IN / AT`, `EVENT_AT`, `DOCUMENT`. Any other order rejected by parser.

### BATCH

```
BEGIN
$x = CREATE NODE "n1" kind = "memory" document = "a"
$y = CREATE NODE "n2" kind = "memory" document = "b"
CREATE EDGE "$x" -> "$y" kind = "next"
COMMIT
```

- `$var` assignment inside BEGIN/COMMIT only
- `$var` refs in later `CREATE EDGE "$x" -> "$y"` statements
- Atomic rollback on any failure
- Do NOT use for bulk - O(cols × nodes) snapshot overhead
- Only write statements allowed inside (not reads)

### SYS

```
SYS STATUS | SYS HEALTH | SYS STATS [NODES|EDGES|MEMORY|WAL]
SYS KINDS | SYS EDGE KINDS | SYS DESCRIBE NODE|EDGE "name"
SYS REGISTER NODE KIND "name" REQUIRED f:t,... [OPTIONAL f:t,...] [EMBED f]
SYS REGISTER EDGE KIND "name" FROM "k1",... TO "k1",...
SYS UNREGISTER NODE|EDGE KIND "name"
SYS SLOW|FREQUENT|FAILED QUERIES [SINCE "t"] [LIMIT n]
SYS EXPLAIN <read_query>    -- MatchQuery/TraverseQuery/NodesQuery: cost plan
                            -- RememberQuery: full dry-run with per-signal candidate scores
SYS CHECKPOINT | SYS REBUILD INDICES
SYS CLEAR LOG|CACHE
SYS WAL STATUS|REPLAY
SYS EXPIRE [WHERE expr]
SYS CONTRADICTIONS [WHERE expr] FIELD f GROUP BY f
SYS SNAPSHOT "name" | SYS ROLLBACK TO "name" | SYS SNAPSHOTS
SYS DUPLICATES [WHERE expr] [THRESHOLD n]
SYS EMBEDDERS | SYS REEMBED
SYS CONNECT [WHERE expr] [THRESHOLD n]              -- auto-wire similar
SYS CONSOLIDATE [THRESHOLD n] [MIN_CLUSTER_SIZE n]  -- cluster + summarise
SYS RETAIN | SYS EVICT [LIMIT n]
SYS OPTIMIZE [COMPACT|STRINGS|EDGES|VECTORS|BLOBS|CACHE]
SYS LOG [WHERE expr | SINCE "t" | TRACE "id"] [LIMIT n]
SYS CRON ADD "name" SCHEDULE "cron" QUERY "dsl"
SYS CRON DELETE|ENABLE|DISABLE|RUN "name"
SYS CRON LIST
SYS EVOLVE RULE "name" WHEN sig OP n [AND ...] THEN action [THEN ...] [COOLDOWN n] [PRIORITY n]
SYS EVOLVE LIST|SHOW|ENABLE|DISABLE|DELETE|RESET "name"
SYS EVOLVE HISTORY [LIMIT n]
```

Evolve actions:
```
SET f = n | SET f = [n,n,n]
ADJUST f BY n [UNTIL n]
ADD f "value"
REMOVE f "value"
RUN ident [ident...]        -- e.g. RUN SYS REEMBED
```

Evolve ops: `>=`, `<=`, `==`, `!=`, `>`, `<`.

### VAULT

```
VAULT NEW "title" [KIND "k"] [TAGS "t"]
VAULT READ "id"
VAULT WRITE "id" SECTION "s" CONTENT "text"
VAULT APPEND "id" SECTION "s" CONTENT "text"
VAULT SEARCH "text" [LIMIT n] [WHERE expr]
VAULT BACKLINKS "id"
VAULT LIST [WHERE expr] [ORDER BY f [ASC|DESC]] [LIMIT n]
VAULT SYNC | VAULT DAILY | VAULT ARCHIVE "id"
```

### WHERE expressions

```
expr     := or_expr
or_expr  := and_expr ("OR" and_expr)*
and_expr := not_expr ("AND" not_expr)*
not_expr := "NOT" not_expr | atom
atom     := field OP value
          | field "CONTAINS" "text"
          | field "LIKE" "pat%"
          | field "IN" (v, v, ...)
          | "SIMILAR" "(" field "," "text" ")" ">" number
          | ("INDEGREE" | "OUTDEGREE") [field] OP number
          | "(" expr ")"

OP    := "=" | "!=" | ">" | ">=" | "<" | "<="
field := IDENT ["." IDENT]            -- dot path allowed
value := STRING | NUMBER | "NULL" | time_expr
time  := "NOW()" | "NOW() - " NUMBER UNIT | "TODAY" | "YESTERDAY"
UNIT  := "s" | "m" | "h" | "d"
```

Examples:

```
kind = "memory" AND importance > 0.5
kind = "message" AND __event_at__ > NOW() - 7d
name LIKE "caroline%"
topic IN ("travel", "food", "sport")
NOT (kind = "entity")
OUTDEGREE > 5
SIMILAR(content, "counseling career") > 0.7
```

### MATCH patterns

```
pattern := step arrow step (arrow step)*
step    := "(" STRING ")"                     -- ("fn_main")  bound node
         | "(" IDENT ["WHERE" expr] ")"       -- (callee)  or  (callee WHERE kind="fn")
arrow   := "-[" expr? "]->"                   -- -[kind="calls"]->  or  -[]->
```

Pattern requires ≥ 1 arrow. Single-step `MATCH ("id")` rejected.

Examples:

```
MATCH ("fn_main") -[kind = "calls"]-> (callee) LIMIT 10
MATCH ("ent:paris") -[kind = "mentions"]-> (msg WHERE kind = "message")
MATCH (session) -[]-> (msg) -[kind = "mentions"]-> (ent WHERE name = "Luna")
```

### REMEMBER fusion (default weights)

```
0.52 × vec_signal      -- max sentence cosine per candidate
0.25 × bm25_signal     -- FTS5 over doc_fts
0.15 × recency_signal  -- exp(-age / half_life), half_life default 7300d
0.08 × graph_signal    -- entity-degree sum, log-scaled (when dsl.graph_signal_enabled)
+ co-occurrence bonus  -- min(vec, bm25) × 0.10 when both fire
+ recall_frequency     -- log1p(recall_count) × 0.05
```

### Per-node signal scores (always populated)

Every REMEMBER result carries these on every returned node:

```
_remember_score  -- fused final; when a reranker ran, this is the rerank score
_vector_sim      -- max sentence cosine across the SQE-expanded query
_bm25_score      -- normalised FTS5 score
_recency_score   -- exp(-age / half_life)
_graph_score     -- normalised entity-degree contribution
_co_bonus        -- min(vec, bm25) * 0.10 when found by BOTH channels
_recall_boost    -- log1p(__recall_count__) * 0.05
_rank_stage      -- "fusion" or "rerank"
_fusion_score    -- pre-rerank fused base (only when a reranker ran)
_rerank_score    -- raw reranker score (only when a reranker ran)
```

### Rich meta["signals"] telemetry

`Result.meta["signals"]` on every REMEMBER (and ANSWER) carries the full pipeline state:

```json
{
  "fusion": {"method": "weighted|rrf", "weights": [...], "graph_signal_enabled": bool},
  "recency": {"half_life_days": 7300.0},
  "sentence_query_expansion": {"enabled": bool, "num_sentences": int},
  "stages": {"gathered_vec": int, "gathered_bm25": int, "union": int,
             "cap_applied": bool, "after_cap": int, "before_rerank": int,
             "final": int},
  "reranker": {"ran": bool, "model": str|null, "error": str|null},
  "nucleus": {"enabled": bool}
}
```

Use it to verify: did BM25 fire (gathered_bm25 > 0)? Did rerank run? Did the candidate cap kick in? Which fusion weights produced this ordering?

### ANSWER verb (REMEMBER + reader LLM)

```
ANSWER "question" [AT "..."] [TOKENS n] [LIMIT n] [WHERE expr] [USING "reader"]
```

Runs REMEMBER internally, hands the top-k passages + question to a reader LLM configured on the SuperGraph, returns `Result(kind="answer", data={"answer": str, "cited_slots": [id], "candidates": [nodes], "reader": str|null}, meta=<REMEMBER signals>)`.

supergraph ships no LLM dependency. The reader is a plain callable the caller wires at construction:

```python
gs = SuperGraph(reader=my_callable)                # default reader
gs = SuperGraph(readers={"fast": a, "careful": b}) # named registry
```

Reader resolution:
1. `USING "name"` -> `readers[name]` (raise if missing)
2. default `reader=`
3. sole entry of `readers` if exactly one
4. else raise `SuperGraphError`

Reader exceptions are caught - `data["error"]` carries the exception message; retrieval state still returned so the caller can inspect.

### SYS EXPLAIN REMEMBER (dry-run)

```
SYS EXPLAIN REMEMBER "text" [AT "..."] [LIMIT n] [WHERE expr]
```

Returns `Result(kind="plan", data={"candidates": [{slot, id, fused_score, vector_sim, bm25_score, recency_score, graph_score, co_bonus, recall_boost}]}, meta={"signals": {...}})`.

Side-effect-free: skips materialization, rerank, nucleus, recall-count bumps. Safe to call repeatedly while tuning fusion weights or diagnosing "why did this rank where it did".

## Escape rules

Inside every `"..."` literal:

- `"` → `\"`
- `\` → `\\`
- No newlines inside DSL strings (Lark newlines separate statements)

Safer: strip newlines from content before emitting. Example for content with quotes:

```
CREATE NODE "msg:1" kind = "message" DOCUMENT "She said \"hello\" and waved."
```

The parser reads `\"` as a literal `"` inside the string. Emitting a bare `"` mid-string ends the literal and breaks the statement.

## Values

| Python/source | DSL literal |
|---|---|
| `"hello"` | `"hello"` |
| `42` or `3.14` | `42` or `3.14` |
| `True` / `False` | `1` / `0` (no boolean keyword) |
| `None` | `NULL` |
| date `2024-03-15` | `"2024-03-15"` (string) |
| NaN, Inf | REJECTED (emit valid numbers only) |

Time expressions in WHERE or EVENT_AT: `NOW()`, `NOW() - 7d`, `TODAY`, `YESTERDAY`. Units: `s`, `m`, `h`, `d`.

## Reserved fields (graph engine)

| Field | Set by | REMEMBER reads |
|---|---|---|
| `__created_at__` | wall clock on CREATE | recency |
| `__updated_at__` | wall clock on UPDATE | recency |
| `__event_at__` | `EVENT_AT "date"` clause | recency (preferred over updated_at) |
| `__confidence__` | `ASSERT ... CONFIDENCE n` | confidence signal |
| `__retracted__` | `RETRACT "id"` | filters retracted from reads |
| `__source__` | `ASSERT ... SOURCE "..."` | provenance |

**Do not** stuff scores into a custom `importance` field expecting REMEMBER to read it. REMEMBER does not read `importance`. Use `ASSERT ... CONFIDENCE n` to set `__confidence__`.

## Gotchas (hard-earned)

**G1. REMEMBER's graph signal ≠ multi-hop.** Boosts chunks mentioning high-degree entities. For real multi-hop, use RECALL.

**G2. BM25 needs `doc_fts` populated.** Only these populate it:
- `INGEST "..."`
- `CREATE NODE ... DOCUMENT "text"` ← cheapest bulk path
Plain `CREATE NODE kind="X" content="..."` without DOCUMENT does NOT populate BM25. LEXICAL returns empty, REMEMBER's BM25 leg is dead.

**G3. Real timestamps can hurt.** 2023 data ingested in 2026 → `exp(-1100/30) ≈ 0` → recency signal dead. For static benchmarks, default wall-clock `now_ms` (uniform recency=1.0) is fine. Only override `__updated_at__` when you KNOW the question-time vs data-time relationship.

**G4. `importance` ≠ `__confidence__`.** REMEMBER reads `__confidence__`. `importance` is a plain column. Set via `ASSERT ... CONFIDENCE n`.

**G5. Do not EMBED entity nodes.** Short names (1-3 tokens) make noisy vectors. Register without EMBED: `SYS REGISTER NODE KIND "entity" REQUIRED name:string`.

**G6. REMEMBER `WHERE kind=...` filters post-gather.** Candidates gathered first, filtered after. Wasteful when non-message kinds have vectors; redundant otherwise.

**G7. `EMBED field` and `DOCUMENT` clause are two paths.** Schema `EMBED content` + `CREATE NODE ... content="..."` → embeds content. `CREATE NODE ... DOCUMENT "text"` (no EMBED) → stores blob + embeds blob. Both set → only EMBED fires, DOCUMENT blob stored but unembedded. Pick one.

**G8. NER duplicate entity names.** Multi-span matches emit same entity twice per message. Naive `CREATE EDGE mentions` loop hits `BatchRollback: Duplicate edge` and rolls back entire BEGIN/COMMIT. Dedupe per-message before emitting edges. Failure mode: bench runs fine for ~100 records then crashes.

**G9. Clause order matters on CREATE NODE / UPSERT NODE.** Grammar requires: `field_pairs` → `VECTOR` → `EXPIRES` → `EVENT_AT` → `DOCUMENT`. Any other order rejected.

**G10. Edges bulk-rebuild on first CSR read.** Interleaving CREATE EDGE with TRAVERSE / MATCH / RECALL triggers O(total_edges) rebuild per interleaving point. Create all edges first, then query.

**G11. `SIMILAR TO NODE "id"` requires node to have a vector.** Otherwise empty result. Check the node's kind was registered with `EMBED`.

**G12. `BEGIN...COMMIT` is atomic + expensive.** Any statement failure rolls back entire block. Not for bulk (snapshot O(cols × nodes)). Use for small atomic groups only (parent + N children, belief sequences).

## Debug: "retrieval looks wrong"

1. BM25 dead? `SELECT COUNT(*) FROM doc_fts` should equal live message count. Zero → use `DOCUMENT` clause or INGEST.
2. Vectors missing? Embedder silently failed or schema EMBED wrong.
3. `__confidence__` unset → flat 1.0 contribution (harmless but uninformative).
4. All `__updated_at__` equal → recency = flat 1.0 (OK for static corpora).
5. Graph-shaped question (multi-entity / cross-session)? Use RECALL + REMEMBER, not REMEMBER alone.
6. WHERE filtered out the answer? Remove filter, retry.
7. Edges unflushed? `TRAVERSE FROM "any" DEPTH 1` - empty → run any no-op query first.
8. Entities contaminating vectors? Remove EMBED from entity kinds.
9. Stuffing scores in `importance`? REMEMBER ignores. Use `__confidence__`.

## Pattern: conversational ingest (LongMemEval / LoCoMo shape)

Per session, emit one statement per line. Adapter parses + executes each.

```
SYS REGISTER NODE KIND "session" REQUIRED session_id:string
SYS REGISTER NODE KIND "message" REQUIRED session:string, role:string OPTIONAL position:int
SYS REGISTER NODE KIND "entity" REQUIRED name:string

CREATE NODE "sess:A1" kind = "session" session_id = "A1" EVENT_AT "2024-03-15"
CREATE NODE "A1:msg0" kind = "message" session = "A1" role = "user" position = 0 EVENT_AT "2024-03-15" DOCUMENT "I have a cat named Luna."
CREATE EDGE "sess:A1" -> "A1:msg0" kind = "has_message"
UPSERT NODE "ent:luna" kind = "entity" name = "Luna"
CREATE EDGE "A1:msg0" -> "ent:luna" kind = "mentions"
CREATE NODE "A1:msg1" kind = "message" session = "A1" role = "assistant" position = 1 EVENT_AT "2024-03-15" DOCUMENT "That\'s lovely! How old is Luna?"
CREATE EDGE "sess:A1" -> "A1:msg1" kind = "has_message"
CREATE EDGE "A1:msg0" -> "A1:msg1" kind = "next"
UPSERT NODE "ent:luna" kind = "entity" name = "Luna"
CREATE EDGE "A1:msg1" -> "ent:luna" kind = "mentions"
```

Rules for emission:
- Schema register statements first (once per run, idempotent).
- Per message: CREATE NODE with DOCUMENT, then `has_message` edge to session, then deduped `mentions` edges.
- Between adjacent messages: `next` edge.
- Dedupe entity names per message (G8). If same entity appears twice in one message's NER output, emit one `mentions` edge.
- Use `UPSERT NODE` for entities (idempotent across sessions; `CREATE NODE` would raise on second occurrence).

## Pattern: belief extraction (ASSERT / RETRACT)

When the LLM spots a fact from conversation:

```
ASSERT "fact:user:pet" kind = "fact" value = "cat" CONFIDENCE 0.85 SOURCE "A1:msg0" EVENT_AT "2024-03-15"
```

When a later message contradicts:

```
RETRACT "fact:user:pet" REASON "message A3:msg5 says user no longer has a cat"
ASSERT "fact:user:pet" kind = "fact" value = "dog" CONFIDENCE 0.92 SOURCE "A3:msg5" EVENT_AT "2024-06-01"
```

Detect contradictions automatically:

```
SYS CONTRADICTIONS FIELD value GROUP BY topic
```

## Pattern: query generation

Given a natural-language question, emit ONE statement that retrieves best. Default to REMEMBER. Switch based on question shape:

| Question shape | Emit |
|---|---|
| "Tell me about X" | `REMEMBER "X" LIMIT 10 WHERE kind = "message"` |
| "Answer X" (caller configured a reader) | `ANSWER "X" LIMIT 10 WHERE kind = "message"` |
| "Everything connected to X" (X is a known entity) | `RECALL FROM "ent:X" DEPTH 2 LIMIT 20` |
| "Find X in date range" | `REMEMBER "X" AT "2024-05" LIMIT 10` |
| "What changed about X" | `REMEMBER "X" LIMIT 10` then check `_retracted_` in results |
| "Count X" | `COUNT NODES WHERE expr` |
| "Contradictions" | `SYS CONTRADICTIONS FIELD value GROUP BY topic` |
| "Which K mention Y" | `MATCH (y WHERE name = "Y") -[]-> (k) LIMIT 10` |
| "Why did this rank where it did" (debug) | `SYS EXPLAIN REMEMBER "X" LIMIT 10` |

## Do / don't

**DO:**
- Emit one statement per line for the adapter to validate + execute
- Schema-register before first CREATE
- Use `DOCUMENT "text"` for conversational messages (one-shot BM25 + vector + blob)
- Dedupe entity mentions per message before emitting edges
- Use `UPSERT NODE` for entities (idempotent)
- Chain adjacent messages with `next` edges
- Escape `"` → `\"` inside every string literal
- Use REMEMBER as default retrieval verb, RECALL for anchored walks

**DON'T:**
- Wrap bulk loads in `BEGIN...COMMIT` (too expensive)
- Expect plain `CREATE NODE kind=... content="..."` to populate BM25
- EMBED entity / short-label nodes
- Override `__updated_at__` unless you have real timestamps
- Emit raw newlines inside string literals
- Stuff scores into `importance` expecting REMEMBER to read them
- Emit invalid grammar orderings (clause order in CREATE NODE matters)
- Use `CREATE NODE` for entities that may already exist - `UPSERT NODE` instead

## Grammar source

Full Lark grammar: `src/supergraph/dsl/grammar.lark` (344 lines). If your emission fails parser validation, diff against this.

Docs: [supergraph-docs.orkait.com/dsl/reference](https://supergraph-docs.orkait.com/dsl/reference).
