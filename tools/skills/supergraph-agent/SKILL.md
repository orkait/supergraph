---
name: supergraph-agent
description: How an autonomous agent should use supergraph as a memory + retrieval substrate. Covers when to write, when to read, how to choose between lexical / semantic / graph queries, and the tradeoffs vs raw context. Pair with the `supergraph-mcp` server for typed tool calls or with `supergraph-dsl` for raw DSL emission.
type: reference
---

# supergraph-agent

You (the agent) have access to a persistent graph + vector store named
**supergraph**. Every fact you store survives across turns and sessions.
Every query is sub-millisecond at small scale and stays cheap as the store
grows. Use it instead of stuffing context.

## When to write (`gs_remember`)

- A user preference, identity, or claim you'd want next session.
- A decision you made and its reasoning, so you can revisit later.
- An external fact you fetched (API result, file content) worth caching.
- A summary of a long thread before context gets compacted.

**Don't store**: passing chitchat, transient state of the current task,
obviously-derivable facts (`2+2=4`).

Each REMEMBER is one node. Multiple sentences -> multiple REMEMBERs.

## When to read

| Need | Tool | Why |
|---|---|---|
| Best general retrieval | `sg_search(query)` | 3-signal fusion (vector + BM25 + recency + graph proximity) - the powerful default |
| "Anything about X?" (fuzzy only) | `gs_similar(text)` | pure embedding distance, robust to phrasing |
| Exact word/phrase match only | `gs_lexical(query)` | pure BM25, no embedding cost |
| You already have a node id | `gs_recall(id, depth)` | + N-hop graph neighbours |
| Walk relationships | `gs_traverse(id, depth)` | full subgraph |
| Question needs LLM synthesis | `sg_answer(query)` | retrieves + runs Bonsai over results (Pro mode for best output) |
| Just want a count | `gs_count_nodes()` | sanity check |

**Default play**: `sg_search` first - the fusion result usually beats
either `gs_lexical` or `gs_similar` alone. Reach for the pure-signal
tools only when you specifically need that ranking discipline. If a
question needs reasoning over multiple facts, `sg_answer`.

## Patterns

### Memory-on-write loop

```
user: "I'm vegetarian and allergic to peanuts."
agent:
  gs_remember("user is vegetarian")
  gs_remember("user is allergic to peanuts")
```

Two facts, two nodes. Future sessions: `gs_similar("dietary needs")` -> both surface.

### Recall-before-respond

Before answering anything user-specific, run a `gs_similar` against the
question. If hits, weave them into the response. If not, ask.

### RAG over your own memory

`sg_answer("what does the user prefer for breakfast?")` will:
1. Embed the query, retrieve top-K nodes by similarity
2. Run the local Bonsai LLM (Pro mode) over the retrieved context
3. Return a grounded answer

No external API call.

## DSL escape hatch

If you need a verb the typed tools don't expose (graph patterns, weighted
shortest path, evolution rules, vault sync, scheduled jobs, etc.), call
`sg_execute("...")` with raw DSL. See the **supergraph-dsl** skill for
the full ~70-verb grammar.

Common ones not in the typed surface:

| Verb | Use |
|---|---|
| `MATCH (a)-[:rel]->(b) WHERE ...` | pattern matching on edges |
| `AGGREGATE NODES GROUP BY tag` | group + count by attribute |
| `SHORTEST PATH FROM "a" TO "b"` | path queries |
| `SYS STATS` | engine diagnostics |
| `VAULT SYNC` | persist to YAML frontmatter |

## Modes

- **Default**: in-process `SuperGraph()`. No HTTP, no playground server.
- **Pro mode** (`SUPERGRAPH_PROFILE=pro`): adds Bonsai 4B local LLM (~33 tok/s on RTX 3060), Jina v3 embedder, TinyBERT NER. Required for high-quality `sg_answer`.
- **Remote** (`SUPERGRAPH_URL=http://host:7200`): forward DSL to a shared playground server. Use only when multiple agents share one store.

The playground HTTP server is **not required** for solo agent use. Default
in-process mode is faster and simpler.

## Anti-patterns

- **Don't dump entire conversation transcripts into one REMEMBER.** Split into atomic facts; the embedder + retriever work better.
- **Don't query before any writes.** Empty store returns 0 — that's not a bug.
- **Don't try to use `sg_answer` without Pro mode.** It will fall back but quality drops sharply.
- **Don't open multiple SuperGraph instances against the same db path concurrently.** Use one MCP server per db, or use remote mode.

## Tradeoffs vs raw context

| | Raw context | supergraph |
|---|---|---|
| Survives session | no | yes |
| Token cost | grows linearly | O(retrieved-K) |
| Cross-session links | manual | automatic via similarity |
| Setup | zero | one MCP server |
| Best for | one-shot tasks | long-running agents, personal assistants |

Use supergraph when context is the bottleneck. Don't use it when the task
fits comfortably in a single turn.
