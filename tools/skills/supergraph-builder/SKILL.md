---
name: supergraph-builder
description: Reference for supergraph's typed Python query builder (`q` / `F` / `P` / `agg` / `Time` / `EvolveWhen` / `EvolveThen`). Use when writing a Python adapter, ingestion loop, or integration that imports supergraph as a library. Covers every builder function with signature + DSL emission, predicate algebra, pattern builder, aggregates, time expressions, evolve rules, plugin registration, and the ingestion patterns that avoid underperforming a plain vector store. For LLM-runtime DSL text emission, use `supergraph-dsl` skill instead.
compatibility: supergraph >= 0.3.0.
metadata:
  author: orkait
  version: "1.0"
---

# supergraph builder

Use the typed Python API for every adapter, ingestion loop, and query call. Import:

```python
from supergraph import q, F, P, agg, Time, EvolveWhen as W, EvolveThen as A
from supergraph import SuperGraph, Query, register_verb
```

`.execute(gs)` runs a Query against a SuperGraph. `.dsl()` returns the compiled string for debugging / tests.

## When

- Writing a Python adapter (benchmark / production / agent integration)
- Ingestion loop importing supergraph
- Query code in a Python service
- Anything where Python typing + IDE autocomplete + injection safety matter

For LLM-runtime text DSL emission, see `supergraph-dsl` skill.

## Why builder over string DSL

- **Injection-safe by construction.** Every user string routed through `dsl_literal`. `q.create_node(id, kind="memory", document=untrusted)` is safe regardless of `untrusted` content.
- **Clause ordering grammar-correct by construction.** Grammar requires EXPIRES before DOCUMENT in CREATE NODE; builder emits them right.
- **Typo / refactor safety.** `q.create_node(id, kind="memory", contnt="...")` is a mypy error. String DSL silently stores an extra column.
- **Parser-roundtrip-verified.** Every emission tested against the real Lark parser.
- **Composable.** `|` batches, `.pipe(fn)` transforms, `F` predicates reusable.

## 3 engines behind 1 DSL

| Engine | Stores | Populated by builder call |
|---|---|---|
| Graph (numpy cols + scipy CSR) | typed fields, reserved cols `__event_at__`, `__confidence__`, `__retracted__`, `__source__` | `q.create_node(...)`, `q.create_edge(...)`, `q.assert_(...)` |
| Vector (usearch HNSW, cosine) | (slot, vector) | `q.create_node(..., document="text")` OR schema `EMBED content` |
| Document (SQLite + FTS5 `doc_fts` + blobs) | BM25 + blob | `q.create_node(..., document="text")` OR `q.ingest(...)` |

One node can live in all three.

## Primitive map

| Query | Hits | Use when |
|---|---|---|
| `q.node(id)` | columns | Know id |
| `q.nodes(...)` | columns | Structured filter |
| `q.similar(text=...)` | vectors | NL cue, no anchor |
| `q.similar(node=...)` | vectors | "Like this one" |
| `q.lexical(text)` | FTS5 | Distinctive keywords |
| `q.remember(text)` | all three + graph signal | Default NL retrieval |
| `q.recall(id, depth=k)` | edges (spreading) | Anchor → connected |
| `q.traverse/path/ancestors/match` | edges (deterministic) | Structured walks |

## Ingestion: 8 rules

1. **Schema first.** `q.sys.register_node_kind(...)` before any `q.create_node`. Pre-allocates numpy dtype. `embed=` marks auto-embed field.
2. **Wrap bulk writes in `gs.deferred_embeddings(batch_size=128)`.** 4-10x speedup on transformer embedders. Neutral on model2vec. Per-call only (not cross-call).
3. **Do NOT use `q.batch(...)` for bulk.** Snapshots all columns for rollback = O(cols × nodes). Use for small atomic groups (parent + N children).
4. **Use `document=` for conversational messages.** One call populates graph row + vector + FTS5 + blob.
5. **Chain messages with `next` edges.** Enables `q.recall`, `q.ancestors`, `q.descendants`.
6. **Build entity graph for cross-session queries.** `q.upsert_node("ent:<slug>", kind="entity", name=name)` + `q.create_edge(msg_id, ent_id, kind="mentions")`. Dedupe per message (G8 below).
7. **Do NOT `embed=` entity nodes.** Short names make noisy vectors.
8. **Create edges in bulk before first read.** Interleaving CREATE EDGE with traversal triggers O(total_edges) CSR rebuild.

## q namespace (builder functions)

Every call returns a `Query`. All keyword args shown; user strings are `dsl_literal`-escaped.

### Reads

| Builder | Emits |
|---|---|
| `q.node(id, *, with_document=False)` | `NODE "id" [WITH DOCUMENT]` |
| `q.nodes(*, kind=None, where=None, limit=None, offset=None, order_by=None)` | `NODES [WHERE ...]` |
| `q.remember(text, *, limit=None, tokens=None, at=None, where=None)` | `REMEMBER "..."` |
| `q.answer(text, *, limit=None, tokens=None, at=None, where=None, using=None)` | `ANSWER "..." [USING "reader"]` |
| `q.recall(from_id, *, depth, limit=None, where=None)` | `RECALL FROM "..." DEPTH n` |
| `q.similar(*, text=None, node=None, vec=None, limit=None, where=None)` | `SIMILAR TO ...` (exactly one target) |
| `q.lexical(text, *, limit=None, where=None)` | `LEXICAL SEARCH "..."` |
| `q.edges(node, *, direction="FROM", where=None, limit=None)` | `EDGES FROM|TO "..."` |
| `q.count_nodes(*, where=None)` | `COUNT NODES` |
| `q.count_edges(*, where=None)` | `COUNT EDGES` |
| `q.traverse(from_id, *, depth, where=None, limit=None)` | `TRAVERSE FROM "..." DEPTH n` |
| `q.subgraph(from_id, *, depth)` | `SUBGRAPH FROM "..."` |
| `q.path(a, b, *, max_depth, where=None)` | `PATH FROM "a" TO "b"` |
| `q.paths(a, b, *, max_depth, where=None)` | `PATHS FROM ...` |
| `q.shortest_path(a, b, *, max_depth=None, where=None)` | `SHORTEST PATH ...` |
| `q.distance(a, b, *, max_depth)` | `DISTANCE ...` |
| `q.weighted_shortest_path(a, b, *, max_depth=None, where=None)` | `WEIGHTED SHORTEST PATH ...` |
| `q.weighted_distance(a, b, *, max_depth=None)` | `WEIGHTED DISTANCE ...` |
| `q.ancestors(id, *, depth, where=None)` | `ANCESTORS OF ...` |
| `q.descendants(id, *, depth, where=None)` | `DESCENDANTS OF ...` |
| `q.common_neighbors(a, b, *, where=None)` | `COMMON NEIGHBORS ...` |
| `q.match(pattern, *, limit=None)` | `MATCH ...` (pattern via `P.*`) |
| `q.what_if_retract(id)` | `WHAT IF RETRACT "id"` |
| `q.aggregate_nodes(*, select, where=None, group_by=None, having=None, order_by=None, order_dir=None, limit=None)` | `AGGREGATE NODES ...` |

### Writes

| Builder | Emits |
|---|---|
| `q.create_node(id, *, kind, event_at=None, expires_in=None, expires_at=None, document=None, vector=None, **fields)` | `CREATE NODE "id" ...` |
| `q.create_node_auto(*, kind, **kwargs)` | `CREATE NODE AUTO ...` (auto-id) |
| `q.update_node(id, **set_fields)` | `UPDATE NODE "id" SET ...` |
| `q.upsert_node(id, *, kind=None, vector=None, expires_in=None, expires_at=None, event_at=None, **fields)` | `UPSERT NODE "id" ...` |
| `q.delete_node(id)` | `DELETE NODE "id"` |
| `q.delete_nodes(*, where)` | `DELETE NODES WHERE ...` (where required) |
| `q.update_nodes(*, where, set)` | `UPDATE NODES WHERE ... SET ...` |
| `q.create_edge(src, tgt, *, kind, **fields)` | `CREATE EDGE "src" -> "tgt" ...` |
| `q.update_edge(src, tgt, *, set, where=None)` | `UPDATE EDGE ...` |
| `q.delete_edge(src, tgt, *, where=None)` | `DELETE EDGE ...` |
| `q.delete_edges(node, *, direction="FROM", where=None)` | `DELETE EDGES ...` |
| `q.increment(id, field_name, *, by)` | `INCREMENT NODE "id" field BY n` |
| `q.assert_(id, *, kind, value=None, confidence=None, source=None, event_at=None, **fields)` | `ASSERT "id" ...` (sets `__confidence__`) |
| `q.retract(id, *, reason=None)` | `RETRACT "id" [REASON ...]` |
| `q.merge(old, into)` | `MERGE NODE "old" INTO "canonical"` |
| `q.propagate(id, *, field, depth)` | `PROPAGATE "id" FIELD f DEPTH n` |
| `q.forget(id)` | `FORGET NODE "id"` (hard delete) |
| `q.connect_node(id, *, threshold=None)` | `CONNECT NODE "id"` |
| `q.ingest(file, *, as_id=None, kind=None, using=None, vision_model=None)` | `INGEST "file"` |
| `q.bind_context(name)` / `q.discard_context(name)` | `BIND/DISCARD CONTEXT "..."` |
| `q.begin()` / `q.commit()` | Literal `BEGIN` / `COMMIT` |
| `q.batch(*stmts)` | `BEGIN ... COMMIT` block |
| `q.var(name, inner)` | `$name = <inner>` (inside batch only) |
| `q.raw(dsl, **params)` | `:name` substitution with `dsl_literal` escape |

### Batch with var refs

```python
q.batch(
    q.var("x", q.create_node("n1", kind="memory", document="a")),
    q.var("y", q.create_node("n2", kind="memory", document="b")),
    q.create_edge("$x", "$y", kind="next"),
).execute(gs)
```

`$x` / `$y` in later statements reference earlier var assignments. Atomic rollback on any failure.

### q.sys namespace

```
q.sys.status() / .health() / .stats(target=None)
q.sys.kinds() / .edge_kinds()
q.sys.describe(entity, name)
q.sys.embedders() / .snapshots()
q.sys.slow_queries(since=None, limit=None)
q.sys.frequent_queries(limit=None)
q.sys.failed_queries(limit=None)
q.sys.explain(query)                              -- query is another Query
q.sys.register_node_kind(name, *, required, optional=None, embed=None)
q.sys.register_edge_kind(name, *, from_kinds, to_kinds)
q.sys.unregister(entity, name)
q.sys.checkpoint() / .rebuild_indices()
q.sys.clear(target)                               -- "LOG" | "CACHE"
q.sys.wal(action)                                 -- "STATUS" | "REPLAY"
q.sys.expire(where=None)
q.sys.contradictions(*, field, group_by, where=None)
q.sys.snapshot(name) / .rollback_to(name)
q.sys.duplicates(where=None, threshold=None)
q.sys.connect(where=None, threshold=None)
q.sys.consolidate(threshold=None, min_cluster_size=None)
q.sys.reembed() / .retain()
q.sys.optimize(target=None)                       -- "COMPACT"|"STRINGS"|"EDGES"|"VECTORS"|"BLOBS"|"CACHE"
q.sys.evict(limit=None)
q.sys.log(where=None, since=None, trace=None, limit=None)
```

`required` / `optional` accept dict `{"f":"string"}` or list `["f:string"]`. `embed=` is a single field name or None.

### q.sys.cron

```
q.sys.cron.add(name, *, schedule, query)   -- query is a string DSL
q.sys.cron.delete(name) / .enable(name) / .disable(name) / .run(name)
q.sys.cron.list()
```

### q.sys.evolve

```
q.sys.evolve.rule(name, *, when, then, cooldown=None, priority=None)
q.sys.evolve.list() / .show(name) / .enable(name) / .disable(name) / .delete(name)
q.sys.evolve.history(limit=None) / .reset()
```

`when` = list of `EvolveCondition` (use `W.cond(...)`). `then` = list of `EvolveAction` (use `A.*`).

### q.vault

```
q.vault.new(title, *, kind=None, tags=None)
q.vault.read(id) / .backlinks(id) / .archive(id)
q.vault.write(id, *, section, content)
q.vault.append(id, *, section, content)
q.vault.search(text, *, limit=None, where=None)
q.vault.list(where=None, order_by=None, limit=None)
q.vault.sync() / .daily()
```

## F: predicate algebra

Immutable. Operators: `&` AND, `|` OR, `~` NOT. `F.true()` / `F.false()` identities. Associative & commutative, auto-flatten.

| Builder | DSL |
|---|---|
| `F.eq(f, v)` | `f = v` |
| `F.ne(f, v)` | `f != v` |
| `F.gt/gte/lt/lte(f, v)` | `f >/>=/</<= v` |
| `F.in_(f, [v1,v2,...])` | `f IN (v1, v2, ...)` |
| `F.not_in(f, [v1,...])` | `NOT (f IN (v1, ...))` |
| `F.contains(f, substr)` | `f CONTAINS "substr"` |
| `F.like(f, "pat%")` | `f LIKE "pat%"` |
| `F.startswith(f, "pfx")` | `f LIKE "pfx%"` |
| `F.is_null(f)` | `f = NULL` |
| `F.is_not_null(f)` | `f != NULL` |
| `F.similar_score(f, text, gt=n)` | `SIMILAR(f, "text") > n` |
| `F.indegree(op, n, field=None)` | `INDEGREE [f] OP n` |
| `F.outdegree(op, n, field=None)` | `OUTDEGREE [f] OP n` |
| `F.raw("custom dsl")` | pass-through (no escape) |
| `F.from_dict({...})` | dict shorthand → F tree |

Dict suffixes: `__gt`, `__lt`, `__gte`, `__lte`, `__ne`, `__in`, `__not_in`, `__contains`, `__startswith`, `__is_null`. Grouping keys: `__and__`, `__or__`, `__not__`.

Values: str/int/float/bool/None/list/tuple/date/datetime/TimeExpr. Bool → 0/1. None → NULL. NaN/Inf rejected at build time.

Examples:

```python
# Simple
F.eq("kind", "memory") & F.gt("importance", 0.5)

# With time
recent = F.gte("__event_at__", Time.now_minus(7, "d"))
q.nodes(where=F.eq("kind", "memory") & recent & ~F.eq("__retracted__", True))

# Dict shorthand
q.nodes(where={"kind": "memory", "importance__gt": 0.5})

# IN, CONTAINS, LIKE
F.in_("topic", ["travel", "food"])
F.contains("content", "urgent")
F.like("name", "caroline%")

# Degree filters
F.outdegree(">", 5)

# Escape hatch
F.raw("custom_fn(x, y) < 100")
```

## P: pattern builder for MATCH

```python
P.node("fn_main")                                # ("fn_main")
P.var("callee")                                  # (callee)
P.var("callee", where=F.eq("kind", "fn"))        # (callee WHERE kind = "fn")

# Chain with .to(next_step, edge=...)
pattern = P.node("fn_main").to(P.var("callee"), edge=F.eq("kind", "calls"))
# ("fn_main") -[kind = "calls"]-> (callee)

q.match(pattern, limit=10).execute(gs)

# Multi-hop
P.node("ent:paris") \
 .to(P.var("msg"), edge=F.eq("kind", "mentions")) \
 .to(P.var("sess"), edge=F.eq("kind", "has_message"))
# ("ent:paris") -[kind = "mentions"]-> (msg) -[kind = "has_message"]-> (sess)
```

Pattern requires ≥ 1 arrow. Single-step MATCH rejected at build time.

## agg + HavingExpr

```python
agg.count()                      # COUNT()
agg.count_distinct("topic")      # COUNT DISTINCT(topic)
agg.sum("importance")
agg.avg("importance")
agg.min("x") / agg.max("x")

# Comparisons build HavingExpr
agg.avg("importance") > 0.5
agg.count() >= 10

# Use in aggregate_nodes
q.aggregate_nodes(
    select=[agg.count(), agg.avg("importance")],
    where=F.eq("kind", "memory"),
    group_by=["topic"],
    having=agg.avg("importance") > 0.5,
    limit=10,
)
```

## Time

```python
Time.now()                       # NOW()
Time.today()                     # TODAY
Time.yesterday()                 # YESTERDAY
Time.now_minus(7, "d")           # NOW() - 7d     (units: s / m / h / d, non-negative int)
```

Use in WHERE via `F.gte("__event_at__", Time.now_minus(7, "d"))`, or as `event_at=Time.today()` on writes.

## EvolveWhen / EvolveThen

```python
from supergraph import EvolveWhen as W, EvolveThen as A

# Conditions (IDENT OP NUMBER; OP in >=, <=, ==, !=, >, <)
W.cond("recall_hit_rate", "<=", 0.4)

# Actions
A.set("target", 1.5)                       # SET target = 1.5
A.set("weights", [0.5, 0.3, 0.2])          # SET weights = [0.5, 0.3, 0.2]
A.adjust("rate", 0.1)                      # ADJUST rate BY 0.1
A.adjust_until("rate", 0.1, 2.0)           # ADJUST rate BY 0.1 UNTIL 2.0
A.add("allow", "kind:memory")              # ADD allow "kind:memory"
A.remove("allow", "kind:test")             # REMOVE allow "kind:test"
A.run("SYS", "REEMBED")                    # RUN SYS REEMBED

q.sys.evolve.rule("r1",
    when=[W.cond("recall_hit_rate", "<=", 0.4)],
    then=[A.run("SYS", "REEMBED")],
    cooldown=86400,
    priority=10,
).execute(gs)
```

## Plugin registration

```python
from supergraph.query import register_verb, Query

@register_verb("ts_downsample")
def ts_downsample(series_id: str, *, window: str, agg: str) -> Query:
    return Query(
        _verb="raw",
        _params={"text": f'TS DOWNSAMPLE "{series_id}" WINDOW "{window}" AGG "{agg}"'},
        _kind="read",
    )

q.ts_downsample("metrics:cpu", window="1h", agg="p95")
```

Attribute lookup on `q` falls through to the plugin registry for names not built in.

## Query composition

```python
# Immutable modifiers on read queries
base = q.nodes(kind="memory")
top10 = base.limit(10)
recent = base.where(F.gt("__event_at__", "2024-01"))   # AND-combines with existing WHERE
combined = base.with_(limit=20, order_by="importance DESC")

# Functional composition with .pipe
def with_recency(q, days):
    return q.where(F.gte("__event_at__", Time.now_minus(days, "d")))

def for_user(q, uid):
    return q.where(F.eq("owner", uid))

q.nodes(kind="memory").pipe(with_recency, days=7).pipe(for_user, uid="u42").limit(10)

# Batch compose with |
b = q.create_node("n1", kind="memory") | q.create_node("n2", kind="memory")
wrapped = q.begin() | b | q.commit()
```

## Debugging

Every `Query` has `.dsl()` - returns the compiled string without executing.

```python
qr = q.remember("question", limit=10, where=F.eq("kind", "message"))
print(qr.dsl())
# REMEMBER "question" LIMIT 10 WHERE kind = "message"
```

Every REMEMBER result exposes per-signal scores on every returned node:

```python
r = q.remember("caroline counseling", limit=1, where=F.eq("kind","message")).execute(gs)
n = r.data[0]
print(n["_remember_score"],    # fused final (or rerank score if a reranker ran)
      n["_vector_sim"],        # max sentence cosine
      n["_bm25_score"],        # normalised FTS5 score
      n["_recency_score"],     # exp(-age / half_life)
      n["_graph_score"],       # entity-degree contribution
      n["_co_bonus"],          # min(vec, bm25) * 0.10 when both fire
      n["_recall_boost"],      # log1p(__recall_count__) * 0.05
      n["_rank_stage"])        # "fusion" or "rerank"
# When a reranker ran, these extras are also set:
#   n["_fusion_score"]   pre-rerank fused base
#   n["_rerank_score"]   raw reranker output
```

`Result.meta["signals"]` carries the rest of the pipeline state:

```python
r.meta["signals"]
# {
#   "fusion": {"method": "weighted", "weights": [...], "graph_signal_enabled": True},
#   "recency": {"half_life_days": 7300.0},
#   "sentence_query_expansion": {"enabled": True, "num_sentences": 1},
#   "stages": {"gathered_vec": N, "gathered_bm25": N, "union": N,
#              "cap_applied": bool, "after_cap": N, "before_rerank": N, "final": N},
#   "reranker": {"ran": bool, "model": str|None, "error": str|None},
#   "nucleus": {"enabled": bool},
# }
```

### Dry-run via `q.sys.explain(inner)`

Pass any `q.remember(...)` Query into `q.sys.explain(...)` to dry-run the pipeline. Skips materialization, rerank, nucleus, and the recall-count bump:

```python
plan = q.sys.explain(q.remember("caroline counseling", limit=3)).execute(gs)
plan.kind            # "plan"
plan.data["candidates"]
# [{"slot": 9, "id": "m1", "fused_score": 0.42,
#   "vector_sim": 0.52, "bm25_score": 0.0, "recency_score": 1.0,
#   "graph_score": 0.0, "co_bonus": 0.0, "recall_boost": 0.0}, ...]
plan.meta["signals"]   # same block as a real REMEMBER
```

Safe to call repeatedly while tuning fusion weights.

### ANSWER: retrieval + reader LLM

`q.answer(...)` runs REMEMBER internally, hands the retrieved passages + question to a configured reader callable, returns an answer with citations. supergraph ships no LLM dependency; the reader is a plain callable:

```python
def my_reader(prompt: str, max_tokens: int = 1000) -> str:
    # call any LLM (openai, litellm, local model, ...)
    ...

gs = SuperGraph(reader=my_reader)

r = q.answer("What is the capital of France?", limit=3).execute(gs)
r.kind           # "answer"
r.data["answer"]        # "Paris"
r.data["cited_slots"]   # ["n0", "n1", "n2"]
r.data["candidates"]    # list of full REMEMBER nodes
r.meta["signals"]       # REMEMBER telemetry
```

Named readers for A/B testing:

```python
gs = SuperGraph(readers={"fast": fast_llm, "careful": careful_llm})
r = q.answer("q", limit=3, using="fast").execute(gs)
r = q.answer("q", limit=3, using="careful").execute(gs)
```

Reader resolution order: `USING name` -> `readers[name]` (raise if missing) -> default `reader=` -> sole entry of `readers` if exactly one -> else `SuperGraphError`.

Reader exceptions are caught, not raised. `data["error"]` carries the exception message; retrieval state (candidates, signals) still returned so the caller can inspect.

## REMEMBER fusion (default)

```
0.52 × vec_signal
0.25 × bm25_signal
0.15 × recency_signal     (half_life default 7300 days via dsl.recency_half_life_days)
0.08 × graph_signal       (when dsl.graph_signal_enabled)
+ co-occurrence  × 0.10
+ recall_frequency × 0.05
```

Config knobs (`supergraph.json` or constructor kwargs):

- `dsl.remember_weights` = `[0.52, 0.25, 0.15, 0.08]` (3 or 4 entries)
- `dsl.fusion_method` = `"weighted"` or `"rrf"`
- `dsl.graph_signal_enabled` = True
- `dsl.recency_half_life_days` = 7300.0
- `dsl.nucleus_expansion` = False
- `vector.search_oversample` = 16

## Gotchas

**G1. REMEMBER's graph signal ≠ multi-hop.** Boosts candidates mentioning high-degree entities. For real multi-hop use `q.recall(...)`.

**G2. BM25 needs `doc_fts` populated.** Only these populate it:
- `q.ingest(...)`
- `q.create_node(..., document="text")` ← cheapest bulk path
Plain `q.create_node(..., content="...")` (no `document=`) does NOT. `q.lexical(...)` returns empty.

**G3. Real timestamps can hurt.** See `supergraph-dsl` G3. Default wall-clock (uniform recency=1.0) is safe for static corpora.

**G4. `importance` ≠ `__confidence__`.** REMEMBER reads `__confidence__`. Set via `q.assert_(id, kind=..., confidence=0.9, ...)`. Plain `importance` field is ignored by REMEMBER.

**G5. Do not `embed=` entity nodes.**

```python
q.sys.register_node_kind("entity", required={"name": "string"}).execute(gs)
# no embed= kwarg
```

**G6. `where=F.eq("kind",...)` on REMEMBER filters post-gather.** Candidates gathered first, filtered after. Redundant if only message-kind has vectors.

**G7. `embed=` schema field and `document=` clause are two paths.** If both set on a create, EMBED fires (DOCUMENT stored but not embedded). Pick one.

**G8. NER duplicate entity names.** Multi-span matches emit same entity twice per message. Deduplicate in Python before emitting `q.create_edge(msg_id, ent_id, kind="mentions")`:

```python
seen: set[str] = set()
for ent in per_msg_entities[i]:
    ent_id = f"ent:{slug(ent)}"
    if ent_id in seen:
        continue
    seen.add(ent_id)
    # emit
```

Failure mode: bench runs fine for ~100 records then crashes with `BatchRollback: Duplicate edge`.

**G9. Clause order is enforced by the builder.** `q.create_node(id, kind, **fields, vector=, expires_in=, event_at=, document=)` is the only shape. Grammar requires `field_pairs → VECTOR → EXPIRES → EVENT_AT → DOCUMENT`. Builder handles ordering automatically.

**G10. Bulk-create edges before first read.** Interleaving `q.create_edge(...).execute(gs)` with `q.traverse(...).execute(gs)` triggers CSR rebuild per interleaving point.

**G11. `q.batch(...)` snapshots all columns for rollback.** Atomic but O(cols × nodes). Not for bulk. Use for small atomic groups (parent + N children, belief sequences).

**G12. `set_reserved` bypasses dirty tracking.** See `supergraph-dsl` G9. For production use `q.update_node(id, __field__=value)` instead of direct numpy writes.

**G13. Single-writer is hard.** Two threads on `gs.execute` where `queued=False` → silent corruption. Use `queued=True` for worker-queue serialisation.

## Pattern A: conversational benchmark ingest

Full LoCoMo / LongMemEval shape using typed builder end-to-end.

```python
from supergraph import SuperGraph, q, F, Time
from supergraph.core.errors import NodeExists

gs = SuperGraph(path=tmpdir, embedder=my_embedder)

# Schema (idempotent - run once)
q.sys.register_node_kind("session", required={"session_id": "string"}).execute(gs)
q.sys.register_node_kind(
    "message",
    required={"session": "string", "role": "string"},
    optional={"position": "int"},
    # No embed= here - messages use document= which goes through the DOCUMENT path
).execute(gs)
q.sys.register_node_kind("entity", required={"name": "string"}).execute(gs)  # no embed=
q.sys.register_edge_kind("has_message", from_kinds=["session"], to_kinds=["message"]).execute(gs)
q.sys.register_edge_kind("next", from_kinds=["message"], to_kinds=["message"]).execute(gs)
q.sys.register_edge_kind("mentions", from_kinds=["message"], to_kinds=["entity"]).execute(gs)

from supergraph.ingest.entity_extract import extract_batch

for session in record.haystack:
    msg_contents = [m.content for m in session.messages]
    per_msg_entities = extract_batch(msg_contents)   # list[list[str]]

    stmts = [q.create_node(f"sess:{session.id}", kind="session", session_id=session.id)]

    for i, msg in enumerate(session.messages):
        msg_id = f"{session.id}:msg{i}"
        stmts.append(q.create_node(
            msg_id, kind="message",
            session=session.id, role=msg.role, position=i,
            document=msg.content,                     # populates BM25 + blob + vector
        ))
        stmts.append(q.create_edge(f"sess:{session.id}", msg_id, kind="has_message"))

        seen: set[str] = set()                        # dedupe per message (G8)
        for ent in per_msg_entities[i]:
            ent_id = f"ent:{slug(ent)}"
            if ent_id in seen:
                continue
            seen.add(ent_id)
            stmts.append(q.upsert_node(ent_id, kind="entity", name=ent))
            stmts.append(q.create_edge(msg_id, ent_id, kind="mentions"))

        if i > 0:
            stmts.append(q.create_edge(
                f"{session.id}:msg{i-1}",
                msg_id,
                kind="next",
            ))

    # Deferred embeddings: batches embedder across all messages in this session
    with gs.deferred_embeddings(batch_size=128):
        for s in stmts:
            s.execute(gs)
```

Query side:

```python
def query(question: str, k: int = 5) -> list[str]:
    depth = 8

    # Hybrid retrieval
    primary = q.remember(
        question, limit=k * depth,
        where=F.eq("kind", "message"),
    ).execute(gs)
    merged: list[str] = [n["content"] for n in primary.data if n.get("content")]

    # Entity-anchored walk for cross-session
    for ent in extract_entities(question)[:3]:
        ent_id = f"ent:{slug(ent)}"
        try:
            rec = q.recall(ent_id, depth=2, limit=k).execute(gs)
            for n in rec.data:
                text = n.get("content", "")
                if text and text not in merged:
                    merged.append(text)
        except Exception:
            pass

    # Recency boost
    recent = q.nodes(
        where=F.eq("kind", "message"),
        order_by="__updated_at__ DESC",
        limit=k * 2,
    ).execute(gs)
    for n in recent.data:
        text = n.get("content", "")
        if text and text not in merged:
            merged.append(text)

    return merged[:k]
```

## Pattern B: document ingest

```python
q.ingest("report.pdf", as_id="doc:q3", kind="report").execute(gs)
q.sys.connect().execute(gs)     # auto-wire similar chunks
```

Tiered router: MarkItDown → PyMuPDF4LLM → Docling → VLM.

## Pattern C: belief tracking

```python
q.assert_(
    "fact:user:pet",
    kind="fact",
    value="cat",
    confidence=0.85,
    source="A1:msg0",
    event_at="2024-03-15",
).execute(gs)

# Later evidence contradicts
q.retract("fact:user:pet", reason="session A3 says user no longer has a cat").execute(gs)
q.assert_(
    "fact:user:pet",
    kind="fact",
    value="dog",
    confidence=0.92,
    source="A3:msg5",
    event_at="2024-06-01",
).execute(gs)

# Detect
q.sys.contradictions(field="value", group_by="topic").execute(gs)
```

## Pattern D: temporal queries

```python
# Recent messages
q.nodes(
    where=F.eq("kind", "message") & F.gt("__updated_at__", Time.now_minus(7, "d")),
    limit=50,
).execute(gs)

# Time-bounded REMEMBER
q.remember("recent concerns", at="2024-05", limit=10).execute(gs)
```

## Pattern E: image / scanned PDF

```python
q.ingest("scan.pdf", using="vision", vision_model="SmolVLM2-2.2B-Instruct-Q4_K_M.gguf").execute(gs)
q.ingest("chart.png", as_id="img:q3", using="vision", vision_model="SmolVLM2-2.2B-Instruct-Q4_K_M.gguf").execute(gs)
```

Pre-pull weights: `supergraph vision serve --pull-only`. Env: `SUPERGRAPH_VISION_MODEL`, `SUPERGRAPH_VISION_URL`.

## Pattern F: audio

```python
q.ingest("interview.mp3").execute(gs)
q.ingest("standup.m4a", as_id="mem:standup-2026-04-15", kind="standup").execute(gs)
```

`[audio]` extra installs faster-whisper (in-process). Default model `base` (~150 MB). Chunks get `[mm:ss-mm:ss]` headings.

## Debug checklist

1. `SELECT COUNT(*) FROM doc_fts` → zero? Use `document=` or `q.ingest(...)`.
2. `gs._vector_store.count()` → matches live message count? Zero → embedder silently failed.
3. `__confidence__` unset → flat 1.0 contribution. Harmless. Set via `q.assert_(..., confidence=n)` if you have real values.
4. All `__updated_at__` equal → recency flat 1.0. OK for static corpora.
5. Graph-shaped question? Combine `q.remember(...)` and `q.recall(...)`.
6. `where=` filtered out the answer? Drop and retry.
7. Edges unflushed? Any `q.traverse(...).execute(gs)` flushes the CSR.
8. Embedder dim mismatch? `q.sys.reembed().execute(gs)`.
9. Entity kinds polluting vectors? Remove `embed=` from their schema.
10. Scores in `importance`? REMEMBER ignores. Use `__confidence__`.

## Do / don't

**DO:**
- Use the builder for every adapter call
- Schema-register with `q.sys.register_node_kind(..., required=..., embed=...)` first
- Wrap bulk writes in `gs.deferred_embeddings(batch_size=128)`
- Use `document=` for conversational messages
- Dedupe entity mentions per message before emitting `q.create_edge(..., kind="mentions")`
- Use `q.upsert_node` for entities (idempotent across sessions)
- Build the entity graph for cross-session questions
- `q.recall(...)` for graph queries, `q.remember(...)` for language queries
- `q.assert_(..., confidence=n)` to set `__confidence__`
- `.dsl()` in tests to assert emitted DSL without executing

**DON'T:**
- Write f-string DSL for user data - always use the builder
- Use `q.batch(...)` for bulk loads
- `embed=` entity / short-label kinds
- Override `__updated_at__` unless you have real timestamps
- Use `q.remember(...)` and wonder why graph edges are ignored
- Stuff scores into `importance` expecting REMEMBER to read them
- Run two writer threads on `queued=False`

## Related

- DSL grammar reference: `supergraph-dsl` skill
- Query builder docs: [supergraph-docs.orkait.com/query-builder](https://supergraph-docs.orkait.com/query-builder)
- Grammar source: `src/supergraph/dsl/grammar.lark`
