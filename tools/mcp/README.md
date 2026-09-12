# supergraph MCP server

Exposes supergraph as agent-callable tools over the Model Context Protocol
(MCP). Use from Claude Desktop, Cursor, or any MCP-aware client to give an
agent a persistent memory + retrieval substrate without writing DSL by hand.

## Why this exists

supergraph already speaks DSL and has a Python API. An agent that wants to
"remember" or "recall" things has three options:

1. Call `gs.execute("REMEMBER \"...\"")` directly in a Python tool.
2. POST to `supergraph playground`'s `/api/execute` over HTTP.
3. Speak MCP and let its host (Claude Desktop, Cursor, etc.) wire the tools
   in for you.

Option 3 needs no playground server, no HTTP, no manual DSL escaping. The
MCP server holds an in-process `SuperGraph()` and translates typed tool
calls into DSL.

## Install

```bash
uv pip install 'supergraphdb[mcp]'
```

This installs the `mcp` Python SDK and registers a `supergraph-mcp`
console script. The server lives inside the wheel at
`supergraph.mcp.server:main` - no need to clone the repo.

## Run

```bash
supergraph-mcp                       # console script (preferred)
python -m supergraph.mcp.server      # equivalent module form
```

Both speak stdio - the transport Claude Desktop / Cursor expect.

## Modes

| Mode | When to use | How |
|---|---|---|
| **In-process** (default) | Single agent, local laptop, no other consumers of the db | nothing - just run the server |
| **Remote** | Multiple agents share one supergraph over HTTP | set `SUPERGRAPH_URL=http://host:7200` and start `supergraph playground` separately |

The in-process default means **the playground HTTP server is NOT required**
for MCP. Keep it disabled unless you want the web UI or multi-agent shared
state.

## Claude Desktop config

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%/Claude/claude_desktop_config.json` (Windows). See
`claude_desktop_config.example.json` in this dir for a copy-paste version.

```json
{
  "mcpServers": {
    "supergraph": {
      "command": "supergraph-mcp",
      "env": {
        "SUPERGRAPH_DB_PATH": "/path/to/supergraph-agent.db"
      }
    }
  }
}
```

For Pro mode (Bonsai LLM + Jina + NER), add `"SUPERGRAPH_PROFILE": "pro"`.

## Tools exposed

| Tool | DSL emitted | Use for |
|---|---|---|
| `gs_remember(text)` | `CREATE NODE AUTO content="..." DOCUMENT "..."` | store one observation (writes + indexes BM25 + embeds) |
| `gs_remember_batch(texts)` | N x `CREATE NODE AUTO ...` | bulk ingest in one round-trip |
| `sg_search(query, limit)` | `REMEMBER "..." LIMIT K` | best general retrieval - 3-signal fusion (vector + BM25 + recency + graph) |
| `gs_recall(node_id, depth, limit)` | `RECALL FROM "..." DEPTH N LIMIT K` | pull a node + neighbours by id |
| `gs_lexical(query, limit)` | `LEXICAL SEARCH "..." LIMIT K` | pure BM25 keyword search |
| `gs_similar(text, limit)` | `SIMILAR TO "..." LIMIT K` | pure vector / fuzzy search |
| `gs_traverse(from_id, depth, limit)` | `TRAVERSE FROM "..." DEPTH N LIMIT K` | walk the graph |
| `sg_answer(query, max_tokens)` | `ANSWER "..." TOKENS K` | RAG over stored content (Pro mode for best results) |
| `gs_count_nodes()` | `COUNT NODES` | check store size |
| `sg_execute(dsl)` | raw DSL | escape hatch for verbs not covered above |

**Note on naming**: the MCP tool `gs_remember` writes; the DSL verb
`REMEMBER` searches (3-signal fusion). The MCP name reflects agent
semantics ("remember this fact" = store it); the DSL verb name reflects
recall semantics ("remember about X" = pull related nodes). The
`sg_search` tool is what wraps the DSL `REMEMBER` verb.

Resource `graph://stats` returns live node count + mode.

All tools catch DSL errors and return `{"error": "..."}` instead of
crashing the transport - the agent can recover and retry.

## Environment

| Var | Default | Meaning |
|---|---|---|
| `SUPERGRAPH_DB_PATH` | `./supergraph-mcp.db` | in-process db location |
| `SUPERGRAPH_PROFILE` | unset | set to `pro` for Bonsai-backed `sg_answer` |
| `SUPERGRAPH_URL` | unset | when set, forwards DSL to a running playground at that URL |
| `SUPERGRAPH_AUTH_TOKEN` | unset | bearer token for the remote URL |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `missing dep 'mcp'` on stderr | install ran without `[mcp]` extra | `uv pip install 'supergraphdb[mcp]'` |
| Tool calls hang on first use | embedder model downloading from HF (~30 MB) | wait for first call to finish; subsequent calls are fast |
| `gs_count_nodes` always returns 0 after `gs_remember` | WAL not yet flushed to nodes table | run `sg_execute("SYS COMMIT")` or wait for periodic flush |
| `sg_answer` returns garbage / errors | not in Pro mode, no Bonsai loaded | set `SUPERGRAPH_PROFILE=pro` and ensure Bonsai gguf is reachable |
| Remote mode 401 | playground requires auth | set `SUPERGRAPH_AUTH_TOKEN` to the value from playground startup logs |

## Related skills

- `tools/skills/supergraph-agent/SKILL.md` - when/why to call these tools
- `tools/skills/supergraph-dsl/SKILL.md` - full DSL grammar (use with `sg_execute`)
- `tools/skills/supergraph-bonsai-dsl/SKILL.md` - LLM-friendly DSL subset
