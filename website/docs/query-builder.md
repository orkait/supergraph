---
title: Query builder
sidebar_position: 4
---

# Query builder

Typed, composable Python API for every verb in the supergraph DSL.

Ships in v0.3.0. Grammar-locked - every builder output is verified to parse. 100% DSL coverage (87 verbs across read / write / SYS / vault / cron / evolve).

```python
from supergraph import SuperGraph, q, F

gs = SuperGraph(path="./brain")

q.create_node("mem:1", kind="memory", topic="travel",
              document="Paris is the capital of France.").execute(gs)

q.remember("European history", limit=10).execute(gs)

q.recall("mem:1", depth=2, limit=20).execute(gs)

q.nodes(where=F.eq("kind", "memory") & F.gt("importance", 0.5), limit=10).execute(gs)

# Full retrieve + synthesize loop (needs SuperGraph(reader=callable))
q.answer("What European capitals have I seen?", limit=5).execute(gs)
```

## Retrieval pipeline (observable by default)

Every `q.remember(...)` result carries per-signal scores on every node (`_remember_score`, `_vector_sim`, `_bm25_score`, `_recency_score`, `_graph_score`, `_co_bonus`, `_recall_boost`, `_rank_stage`) plus a rich `Result.meta["signals"]` block describing fusion weights, recency half-life, per-stage candidate counts, reranker state, and nucleus state.

Dry-run the pipeline with `q.sys.explain(...)` to see the same telemetry without mutating state or calling the reranker:

```python
plan = q.sys.explain(q.remember("European capitals", limit=3)).execute(gs)
plan.data["candidates"]   # [{slot, id, fused_score, vector_sim, ...}, ...]
plan.meta["signals"]      # full pipeline telemetry
```

## Retrieval + reader synthesis

`q.answer(...)` runs `REMEMBER` internally, hands retrieved passages + question to a user-supplied reader LLM, returns an answer with citations. supergraph ships no LLM dependency; the reader is a plain callable wired at `SuperGraph(reader=...)` or `SuperGraph(readers={"name": callable})`. Reader exceptions are captured in `data["error"]` rather than raised.

```python
def my_reader(prompt: str, max_tokens: int = 1000) -> str:
    # call any LLM backend (openai, litellm, local, ...)
    ...

gs = SuperGraph(reader=my_reader)
r = q.answer("What is the capital of France?", limit=3).execute(gs)

r.data["answer"]         # "Paris"
r.data["cited_slots"]    # ["mem:paris", "mem:eiffel", ...]
r.data["candidates"]     # full REMEMBER nodes with per-signal scores
r.meta["signals"]        # REMEMBER pipeline telemetry
```

Use named readers to A/B two LLMs on the same query:

```python
gs = SuperGraph(readers={"fast": fast_llm, "careful": careful_llm})
q.answer("q", limit=3, using="fast").execute(gs)
q.answer("q", limit=3, using="careful").execute(gs)
```

## Why use the builder

- **Escape-safe by construction.** User-supplied strings can't break out of DSL slots. `q.nodes(kind=user_input)` is injection-proof.
- **IDE autocomplete.** Every verb is a function. Misspelled kwargs fail at lint time.
- **Composable.** Predicate algebra via `F`, clone-with-override via `.limit/.where/.with_`, reusable transformations via `.pipe()`, batch compose via `|`.
- **Debuggable.** `q.verb(...).dsl()` returns the compiled string any time.
- **Grammar-locked.** Every builder is parser-roundtrip-tested; drift is caught in CI.

Compare:

```python
# Before: string interpolation (injection risk, typo fragile)
gs.execute(f'CREATE NODE "{id}" kind = "memory" DOCUMENT "{untrusted_text}"')

# After: builder (escape-safe, IDE-validated, composable)
q.create_node(id, kind="memory", document=untrusted_text).execute(gs)
```

## Import surface

```python
from supergraph import q, F, Query, register_verb
```

- `q` - namespace holding every built-in verb. `q.nodes(...)`, `q.sys.status()`, `q.vault.search(...)`.
- `F` - predicate algebra for WHERE clauses.
- `Query` - return type of every builder. Has `.dsl()`, `.execute(gs)`, modifiers, `.pipe()`, `|` compose.
- `register_verb` - decorator for third-party packages to register custom verbs under `q.<name>`.

## Composability primitives

### Predicate algebra (`F`)

First-class filter objects. Immutable. Operators: `&`, `|`, `~`.

```python
travel    = F.eq("topic", "travel")
recent    = F.gte("__event_at__", "2024-01")
important = F.gt("importance", 0.5)
live      = ~F.eq("__retracted__", True)

q.nodes(where=travel & recent & important & live, limit=10)
```

Dict shorthand compiles to `F` internally:

```python
q.nodes(where={"topic": "travel", "importance__gt": 0.5})
# equivalent to
q.nodes(where=F.eq("topic", "travel") & F.gt("importance", 0.5))
```

Supported ops: `eq` (default), `ne`, `gt`, `gte`, `lt`, `lte`, `in_`, `not_in`, `startswith`, `contains`, `is_null`, `is_not_null`, `raw`.

Dict shorthand recognises `field__op` suffix (`importance__gt`) plus `__and__`, `__or__`, `__not__` grouping keys.

Escape hatch: `F.raw("distance(x, y) < 100")`.

### Immutable modifiers

Read queries chain modifiers; base is never mutated.

```python
base  = q.nodes(kind="memory")        # NODES WHERE kind = "memory"
top10 = base.limit(10)
high  = base.where(F.gt("importance", 0.5))   # AND-combines
custom = base.with_(limit=20, order_by="importance DESC")
```

Available modifiers on read queries: `.limit(n)`, `.where(pred)`, `.tokens(n)` (REMEMBER), `.at(date)` (REMEMBER), `.order_by(expr)`, `.with_(**kw)` (atomic replace of any named modifier).

### Functional composition (`.pipe()`)

Any Python function taking a `Query` and returning a `Query` is reusable.

```python
def with_recency(q, days): return q.where(F.gte("__event_at__", f"2024-{days}"))
def for_user(q, uid):      return q.where(F.eq("owner", uid))

q.nodes(kind="memory").pipe(with_recency, days=7).pipe(for_user, uid="u42").limit(10)
```

### Batch compose (`|`)

Two statements become a `BEGIN..COMMIT` block via `q.begin() | ... | q.commit()`, or shorthand `q.batch(...)`.

```python
q.batch(
    q.create_node("n1", kind="memory", document="x"),
    q.create_node("n2", kind="memory", document="y"),
    q.create_edge("n1", "n2", kind="next"),
).execute(gs)
```

### Raw escape hatch

When a verb is not yet covered (or you need custom grammar), `q.raw` does parameter-safe interpolation.

```python
q.raw('CREATE NODE :id kind = :k DOCUMENT :doc',
      id="mem:1", k="memory", doc=untrusted_text).execute(gs)
```

Missing or extra params raise at build time.

## Verb reference (87 verbs)

### Reads (23)

| Builder | DSL emission |
|---|---|
| `q.node(id, with_document=False)` | `NODE "id" [WITH DOCUMENT]` |
| `q.nodes(kind?, where?, limit?, offset?, order_by?)` | `NODES [WHERE ...] [ORDER BY ...] [LIMIT n] [OFFSET n]` |
| `q.edges(node, direction="FROM", where?, limit?)` | `EDGES FROM "..." [WHERE ...] [LIMIT n]` |
| `q.traverse(from_id, depth, where?, limit?)` | `TRAVERSE FROM "..." DEPTH n ...` |
| `q.subgraph(from_id, depth)` | `SUBGRAPH FROM "..." DEPTH n` |
| `q.path(a, b, max_depth, where?)` | `PATH FROM "a" TO "b" MAX_DEPTH n` |
| `q.paths(a, b, max_depth, where?)` | `PATHS FROM "a" TO "b" MAX_DEPTH n` |
| `q.shortest_path(a, b, max_depth?, where?)` | `SHORTEST PATH FROM "a" TO "b" ...` |
| `q.distance(a, b, max_depth)` | `DISTANCE FROM "a" TO "b" MAX_DEPTH n` |
| `q.weighted_shortest_path(a, b, max_depth?, where?)` | `WEIGHTED SHORTEST PATH ...` |
| `q.weighted_distance(a, b, max_depth?)` | `WEIGHTED DISTANCE ...` |
| `q.ancestors(id, depth, where?)` | `ANCESTORS OF "id" DEPTH n` |
| `q.descendants(id, depth, where?)` | `DESCENDANTS OF "id" DEPTH n` |
| `q.common_neighbors(a, b, where?)` | `COMMON NEIGHBORS OF "a" AND "b"` |
| `q.match(pattern, limit?)` | `MATCH <pattern>` |
| `q.count_nodes(where?)` / `q.count_edges(where?)` | `COUNT NODES/EDGES [WHERE ...]` |
| `q.aggregate_nodes(select, where?, group_by?, having?, order_by?, order_dir?, limit?)` | `AGGREGATE NODES ... SELECT ...` |
| `q.recall(from_id, depth, limit?, where?)` | `RECALL FROM "..." DEPTH n ...` |
| `q.similar(text? / node? / vec?, limit?, where?)` | `SIMILAR TO <target> ...` |
| `q.lexical(text, limit?, where?)` | `LEXICAL SEARCH "..." ...` |
| `q.remember(text, limit?, tokens?, at?, where?)` | `REMEMBER "..." [AT ...] [TOKENS n] [LIMIT n] [WHERE ...]` |
| `q.answer(text, limit?, tokens?, at?, where?, using?)` | `ANSWER "..." [AT ...] [TOKENS n] [LIMIT n] [WHERE ...] [USING "reader"]` |
| `q.what_if_retract(id)` | `WHAT IF RETRACT "id"` |

### Writes (20) + Control (3)

| Builder | DSL emission |
|---|---|
| `q.create_node(id, kind, **fields, event_at?, expires_in?, document?, vector?)` | `CREATE NODE "..." kind = "..." [fields] [VECTOR] [EXPIRES IN n<smhd>] [EVENT_AT ...] [DOCUMENT ...]` |
| `q.update_node(id, **set)` | `UPDATE NODE "..." SET ...` |
| `q.upsert_node(id, kind?, **fields, vector?, expires_in?, event_at?)` | `UPSERT NODE "..." ...` |
| `q.delete_node(id)` | `DELETE NODE "id"` |
| `q.delete_nodes(where=)` | `DELETE NODES WHERE ...` (where required) |
| `q.update_nodes(where=, set=)` | `UPDATE NODES WHERE ... SET ...` |
| `q.create_edge(src, tgt, kind, **fields)` | `CREATE EDGE "src" -> "tgt" kind = "..."` |
| `q.update_edge(src, tgt, set=, where?)` | `UPDATE EDGE "src" -> "tgt" SET ... [WHERE ...]` |
| `q.delete_edge(src, tgt, where?)` | `DELETE EDGE "src" -> "tgt" [WHERE ...]` |
| `q.delete_edges(node, direction="FROM", where?)` | `DELETE EDGES FROM/TO "..." [WHERE ...]` |
| `q.increment(id, field, by=)` | `INCREMENT NODE "id" field BY n` |
| `q.assert_(id, kind, value?, confidence?, source?, event_at?, **fields)` | `ASSERT "id" kind = "..." value = ... [CONFIDENCE n] [SOURCE ...] [EVENT_AT ...]` |
| `q.retract(id, reason?)` | `RETRACT "id" [REASON ...]` |
| `q.merge(old, into)` | `MERGE NODE "old" INTO "into"` |
| `q.propagate(id, field, depth)` | `PROPAGATE "id" FIELD field DEPTH n` |
| `q.forget(id)` | `FORGET NODE "id"` |
| `q.connect_node(id, threshold?)` | `CONNECT NODE "id" [THRESHOLD n]` |
| `q.ingest(file, as_id?, kind?, using?, vision_model?)` | `INGEST "file" [AS ...] [KIND ...] [USING <parser> or USING VISION "model"]` |
| `q.bind_context(name)` / `q.discard_context(name)` | `BIND / DISCARD CONTEXT "..."` |
| `q.begin()` / `q.commit()` / `q.batch(stmts...)` | `BEGIN ... COMMIT` |

### SYS (29) + Cron (6) + Evolve (8)

Full list under `q.sys.*`: `status, stats(target?), health, kinds, edge_kinds, describe(entity, name), embedders, snapshots, slow_queries, frequent_queries, failed_queries, explain(inner), register_node_kind, register_edge_kind, unregister, checkpoint, rebuild_indices, clear(target), wal(action), expire(where?), contradictions(field, group_by, where?), snapshot(name), rollback_to(name), duplicates(where?, threshold?), connect(where?, threshold?), consolidate(threshold?, min_cluster_size?), reembed, retain, optimize(target?), evict(limit?), log(where? | since? | trace?, limit?)`.

Cron: `q.sys.cron.{add(name, schedule, query), delete(name), enable(name), disable(name), list(), run(name)}`.

Evolve: `q.sys.evolve.{rule(name, when, then, cooldown?, priority?), list(), show(name), enable(name), disable(name), delete(name), history(limit?), reset()}`.

### VAULT (10)

`q.vault.{new(title, kind?, tags?), read(id), write(id, section, content), append(id, section, content), search(text, limit?, where?), backlinks(id), list(where?, order_by?, limit?), sync(), daily(), archive(id)}`.

## Plugin registration

Third-party packages can register custom verbs. v1 ships the registry but does not use it internally.

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

Attribute lookup on `q` falls through to the registry if the requested name is not a built-in verb.

## Testing the builder in your code

Every `Query` has `.dsl()`. Use it in your tests to assert the exact DSL your code emits without hitting a SuperGraph:

```python
def test_my_adapter_emits_right_query():
    out = my_adapter.build_query(user_input="...").dsl()
    assert "REMEMBER" in out
    assert "LIMIT 10" in out
```

## Related

- [Grammar source](https://github.com/orkait/supergraph/blob/main/src/supergraph/dsl/grammar.lark) - source of truth for DSL syntax.
- [DSL reference](./dsl/reference) - every verb in the DSL.
