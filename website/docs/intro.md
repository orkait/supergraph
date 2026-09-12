---
slug: /
title: Introduction
sidebar_position: 1
---

# supergraph

Memory infrastructure for AI agents. Nodes and edges with a typed DSL. Retrieve by meaning, association, text, or any mix - one call. Runs in-process, persists to SQLite. No server.

## 60-second start

```bash
pip install supergraphdb
```

```python
from supergraph import SuperGraph

g = SuperGraph(path="./brain")

g.execute('CREATE NODE "mem:paris" kind = "memory" '
          'DOCUMENT "Paris is the capital of France, famous for the Eiffel Tower."')
g.execute('CREATE NODE "mem:rome" kind = "memory" '
          'DOCUMENT "Rome is the capital of Italy, home to the Colosseum."')
g.execute('CREATE EDGE "mem:paris" -> "mem:rome" kind = "both_european_capitals"')

g.execute('REMEMBER "European history" LIMIT 5')
g.execute('RECALL FROM "mem:paris" DEPTH 2 LIMIT 10')
g.execute('LEXICAL SEARCH "Eiffel Tower" LIMIT 5')
g.execute('SIMILAR TO "capital city" LIMIT 5')

g.close()
```

## Typed query builder

Every DSL verb is a typed Python function. Escape-safe, autocomplete-friendly, composable.

```python
from supergraph import q, F, Time

q.create_node("mem:paris", kind="memory",
              document="Paris is the capital of France.").execute(g)

q.nodes(
    where=F.eq("kind", "memory")
          & F.gt("importance", 0.5)
          & F.gte("__event_at__", Time.now_minus(7, "d")),
    limit=10,
).execute(g)
```

87 typed verbs. 100% DSL coverage. 100% line coverage. Injection-proof via a single escape helper. Full reference: [Query builder](./query-builder).

## Why supergraph

Most agent memory is a vector DB wrapper. Fine for simple lookup, breaks on:

- **Multi-signal retrieval** - vectors miss keyword matches, BM25 misses semantic matches. You need both, plus graph structure and recency, fused.
- **Graph-native ops** - spreading activation, subgraph extraction, path queries, counterfactuals. First-class DSL, not a bolt-on.
- **Temporal awareness** - `__event_at__` is a reserved column, not a convention.
- **Belief tracking** - ASSERT with confidence, RETRACT when wrong, find CONTRADICTIONS automatically.
- **Zero infra** - SQLite + numpy + usearch. No Docker, no server, no cloud.

## Next steps

- [Installation](./installation) - core + optional extras
- [Architecture](./concepts/architecture) - three engines, one DSL
- [REMEMBER pipeline](./concepts/remember-pipeline) - retrieval internals
- [Query builder](./query-builder) - typed Python API
- [DSL reference](./dsl/reference) - every verb
- [First memory walkthrough](./guides/first-memory) - end-to-end on LoCoMo
- [Benchmarks](./benchmarks/overview) - LongMemEval + LoCoMo results
