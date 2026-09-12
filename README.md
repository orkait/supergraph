<div align="center">

# supergraph

**A memory database for AI agents**

[![CI](https://github.com/orkait/supergraph/actions/workflows/ci.yml/badge.svg)](https://github.com/orkait/supergraph/actions/workflows/ci.yml)
[![Package](https://img.shields.io/badge/pip-supergraphdb-f59e0b?logo=pypi&logoColor=white)](https://pypi.org/project/supergraphdb/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-ea580c?logo=gnu&logoColor=white)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-website%2Fdocs-f59e0b?logo=readthedocs&logoColor=white)](website/docs/intro.md)

</div>

---

An embedded memory database for AI agents. Facts get written with confidence scores, expire, get contradicted, decay by recency. Retrieval fuses vector similarity, BM25, graph structure, and recency in one call. Everything goes through a typed DSL. Runs in-process, persists to SQLite.

Status: v0.7.0, alpha. Two parts: this substrate (`import supergraph`) and the [superclaw](superclaw/README.md) terminal agent that uses it as memory.

## 📦 Install

```bash
pip install supergraphdb
```

The distribution is `supergraphdb`; the import package is `supergraph`. The
plain name was already taken on PyPI by an unrelated project.

Core ships with [model2vec](https://github.com/MinishLab/model2vec) as the default embedder. Swap for Jina v5, bge-*, EmbeddingGemma, or any ONNX / GGUF model via `supergraph install-embedder`. PDFs, images, audio, GPU, and the web UI are opt-in extras.

```bash
pip install 'supergraphdb[ingest]'       # PDF / DOCX / HTML
pip install 'supergraphdb[vision]'       # local VLM for images + scanned PDFs
pip install 'supergraphdb[audio]'        # faster-whisper speech-to-text
pip install 'supergraphdb[playground]'   # FastAPI web UI
pip install 'supergraphdb[gpu]'          # onnxruntime-gpu, Linux x86_64, CUDA 12
pip install 'supergraphdb[mcp]'          # Model Context Protocol server (supergraph-mcp)
pip install 'supergraphdb[pro]'          # one-shot agentic memory bundle (see Pro mode below)
pip install 'supergraphdb[superclaw]'    # the superclaw terminal coding agent (TUI + headless exec)
```

Full extras matrix: [Installation](website/docs/installation.md).

## 🚀 Quickstart

```python
from supergraph import SuperGraph

g = SuperGraph(path="./brain")

g.execute('CREATE NODE "mem:paris" kind = "memory" '
          'DOCUMENT "Paris is the capital of France, famous for the Eiffel Tower."')
g.execute('CREATE NODE "mem:rome" kind = "memory" '
          'DOCUMENT "Rome is the capital of Italy, home to the Colosseum."')
g.execute('CREATE EDGE "mem:paris" -> "mem:rome" kind = "both_european_capitals"')

g.execute('REMEMBER "European history" LIMIT 5')          # hybrid fusion
g.execute('RECALL FROM "mem:paris" DEPTH 2 LIMIT 10')     # graph walk
g.execute('LEXICAL SEARCH "Eiffel Tower" LIMIT 5')        # BM25
g.execute('SIMILAR TO "capital city" LIMIT 5')            # vector only
```

`DOCUMENT "text"` populates the vector index, FTS5 index, and blob storage in one shot. Without it, a node is structured data only.

## 🌳 Natural-language ingest (Bonsai)

For agent-conversation memory, writing DSL by hand is the wrong abstraction. supergraph ships `BonsaiIngestor`, an LLM-driven NL→DSL converter built on a 4B Ternary-Bonsai GGUF (1.1 GB, runs on CPU at ~20 tok/s, ~150 tok/s on a CUDA 12 GPU). It reads natural-language turns and emits the DSL statements that mirror them.

```python
from supergraph import SuperGraph
from supergraph.bonsai_ingestor import BonsaiIngestor, _DEFAULT_LITE_PROMPT_PATH

g = SuperGraph(path="./brain")
ing = BonsaiIngestor(
    model_path="./models/Ternary-Bonsai-4B-TQ1_0.gguf",
    gs=g,
    skill_path=str(_DEFAULT_LITE_PROMPT_PATH),  # or omit for the full prompt
    n_gpu_layers=-1,                             # 0 for CPU; -1 to offload all
)

ing.ingest("Kailash joined OpenAI.",     msg_id="m1")  # @UPSERT/@UPSERT/@EDGE
ing.ingest("I prefer tea to coffee.",    msg_id="m2")  # @BELIEF
ing.ingest("Maria moved to Berlin.",     msg_id="m3")
ing.ingest("Actually I drink coffee now.", msg_id="m4")  # @RETRACT + @BELIEF

# Retrieval is the same NL surface. Question-shaped turns emit @ANSWER /
# @REMEMBER / @SIMILAR / @LEXICAL / @RECALL / @PATH depending on phrasing.
res = ing.ingest("Where does Maria work?", msg_id="q1")          # executes @ANSWER (needs reader=)
preview = ing.ingest("Where does Maria work?", msg_id="q1", dry_run=True)  # returns DSL only
```

`dry_run=True` returns the synthesized DSL without touching the store - useful for previewing or building training data, but it does NOT produce an answer. To get the answer, leave `dry_run=False` (the default) and configure a reader on the `SuperGraph` (see [ANSWER](#-answer-retrieval--reader-llm) below).

Prompt variants:
- `bonsai_dsl_prompt_lite.txt` (~600 system tokens, 16 verbs, ingest+retrieval): production sweet spot.
- `bonsai_dsl_prompt.txt` (~1700 system tokens, ~50 verbs, all admin DSL): full control surface.

Persistent KV cache (`kv_cache_path=...`) cuts cold start from ~10 s to ~1 s across process restarts.

## 🏗️  Architecture

<p align="center">
  <img src="website/static/img/architecture.svg" alt="supergraph architecture: DSL + three storage engines + ingest pipeline + retrieval" width="760">
</p>

Three engines behind one DSL.

- **Graph**: columnar numpy arrays + scipy CSR edge matrices. Reserved columns `__event_at__`, `__confidence__`, `__retracted__`, `__source__` are first-class.
- **Vector**: usearch HNSW, cosine. Auto-embedding on `DOCUMENT` or `EMBED content` schemas.
- **Document**: SQLite + FTS5 for BM25 and blobs. Single-owner advisory lock on the path.

The **DSL** is Lark LALR(1). Every write, read, `INGEST`, and `SYS *` goes through it.

Deep dive: [Architecture](website/docs/concepts/architecture.md) · [Edge matrix](website/docs/concepts/edge-matrix.md).

The two parts of the project - this substrate and the `superclaw` harness - are mapped in [ARCHITECTURE.md](ARCHITECTURE.md).

## 🧠 REMEMBER

`REMEMBER` fuses four signals at retrieval time. `SIMILAR`, `LEXICAL`, `RECALL` each expose a single leg.

<p align="center">
  <img src="website/static/img/remember.svg" alt="REMEMBER 5-stage retrieval pipeline" width="620">
</p>

| Signal | Default weight | Source |
|---|---|---|
| `vec_signal` | 0.52 | max sentence cosine over usearch ANN |
| `bm25_signal` | 0.25 | SQLite FTS5 over `doc_fts` |
| `recency` | 0.15 | `exp(-age / half_life)` from `__event_at__` |
| `graph_signal` | 0.08 | sum of entity degrees |

Weights are configurable via `supergraph.json`, `SUPERGRAPH_DSL_*` env vars, or constructor kwargs.

Every result returns per-signal scores on every node and a `meta["signals"]` block with the full pipeline state (fusion weights, per-stage candidate counts, reranker status):

```python
r = g.execute('REMEMBER "Caroline counseling" LIMIT 1 WHERE kind = "message"')
n = r.data[0]
print(n["_remember_score"], n["_vector_sim"], n["_bm25_score"],
      n["_recency_score"], n["_graph_score"], n["_co_bonus"],
      n["_recall_boost"], n["_rank_stage"])
r.meta["signals"]  # {fusion, recency, stages, reranker, nucleus, ...}
```

Dry-run the pipeline without mutating recall counts:

```python
g.execute('SYS EXPLAIN REMEMBER "Caroline counseling" LIMIT 3')
# kind="plan", candidates with per-signal scores, full meta["signals"]
```

Deep dive: [REMEMBER pipeline](website/docs/concepts/remember-pipeline.md).

## 💬 ANSWER (retrieval + reader LLM)

For a full retrieve + synthesize loop, wire a reader callable and use `ANSWER`:

```python
def my_reader(prompt: str, max_tokens: int = 1000) -> str:
    ...  # call any LLM (openai, litellm, local, ...)

g = SuperGraph(path="./brain", reader=my_reader)

r = g.execute('ANSWER "What is the capital of France?" LIMIT 3')
r.data["answer"]         # "Paris"
r.data["cited_slots"]    # ["mem:paris", ...]
r.meta["signals"]        # same telemetry as REMEMBER
```

supergraph ships no LLM dependency. The reader is a plain callable; bring your own. Named readers (`SuperGraph(readers={"fast": a, "careful": b})`) enable A/B via `ANSWER "q" USING "careful"`.

A wall-clock `reader_timeout_seconds` (default 60s) bounds every reader invocation. On timeout the result still comes back, with `data["answer"] == ""`, `data["error"]` describing the timeout, and a `meta["warnings"]` entry — the executor never blocks indefinitely on a hung LLM.

```python
SuperGraph(reader=slow_local_llm, reader_timeout_seconds=120.0)
```

## 🪶 Python sugar: `g.ask(...)` and `g.write(...)`

Two thin wrappers for agent apps that don't want to hand-write DSL:

```python
from supergraph import SuperGraph
from supergraph.bonsai_ingestor import BonsaiIngestor

g = SuperGraph(
    path="./store",
    reader=my_reader,
    ingestor=lambda gs: BonsaiIngestor(gs=gs),     # auto-fetches Bonsai GGUF on first use
)

g.write("Maria moved to Berlin.", msg_id="m1")     # NL turn -> @UPSERT/@EDGE -> DSL -> execute
g.write("I prefer tea to coffee.", msg_id="m2")    # NL turn -> @BELIEF -> DSL -> execute

answer = g.ask("Where does Maria work?", limit=5)  # ANSWER DSL with proper escaping
print(answer.data["answer"], answer.data["cited_slots"])
```

- **`g.ask(question, *, limit=None, using=None) -> Result`** — equivalent to `execute('ANSWER "..." [LIMIT n] [USING "name"]')` with quote/backslash escaping handled. Requires a `reader=` callable.
- **`g.write(text, *, msg_id, session_id="default", role="user", dry_run=False)`** — NL turn through the configured `ingestor=` factory. The factory is called lazily on first invocation so the ingestor receives a fully-initialised `SuperGraph`. Without the factory, `write()` raises with a copy-pasteable example.

For graph-aware retrieval (`@REMEMBER`/`@SIMILAR`/`@LEXICAL`/`@RECALL`/`@PATH`), drive the ingestor directly via `g.write(question, msg_id=..., dry_run=True)` and inspect the synthesized DSL — the question-shape routing lives in the Bonsai prompt.

## 🐍 Typed query builder

Every DSL verb has a typed function. Same grammar, IDE autocomplete, injection-safe.

```python
from supergraph import q, F, Time

q.create_node("mem:paris", kind="memory",
              document="Paris is the capital of France.").execute(g)

recent = F.gte("__event_at__", Time.now_minus(7, "d"))
q.nodes(where=F.eq("kind", "memory") & recent & ~F.eq("__retracted__", True))

q.batch(
    q.var("x", q.create_node("n1", kind="memory", document="a")),
    q.var("y", q.create_node("n2", kind="memory", document="b")),
    q.create_edge("$x", "$y", kind="next"),
).execute(g)
```

Full reference: [Query builder](website/docs/query-builder.md).

## 📊 Benchmarks

**LongMemEval-S**, 500 records, Jina v5 Small 1024d, Kaggle T4 GPU, 2026-04-19. Public kernel: [kaggle.com/code/superkaiii/supergraph-jina-v5-small](https://www.kaggle.com/code/superkaiii/supergraph-jina-v5-small).

| Overall | knowledge-update | single-session-assistant | single-session-user | multi-session | temporal | preference |
|---|---|---|---|---|---|---|
| **97.0%** | 100.0% | 100.0% | 98.6% | 98.5% | 94.7% | 83.3% |

Query p50 46 ms / p95 76 ms. Retrieval-only, no LLM judge.

**LoCoMo**, conv-26 199Q full, jina-v5-small 1024d, MiniMax M2.7 reader. Best adapter (Bonsai NL→DSL ingest) overall token-F1 **0.476** vs 0.464 deterministic NER + 0.392 remote-LLM ingest, all on the same retrieval stack. Scoring matched byte-for-byte against snap-research/locomo `task_eval/evaluation.py` (parity test in `tests/test_locomo_scoring_parity.py`).

Full methodology: [Benchmarks](website/docs/benchmarks/overview.md).

## ⚡ GPU offload (opt-in, off by default)

supergraph never grabs a GPU implicitly. Every `*_gpu_layers` default is 0 (CPU). To opt in, install `[gpu]` (and a CUDA-built `llama-cpp-python` wheel for Bonsai/embedder/reranker) and call `gpu.setup()`:

```python
from supergraph import gpu
status = gpu.setup()
print(status.ready, status.provider, status.device_name, status.error)
```

`gpu.setup()` is idempotent and does the dirty work: discovers any `nvidia-*-cu12` pip wheels under `site-packages`, ctypes-preloads their `.so` files in dependency order so `LD_LIBRARY_PATH` doesn't have to be set externally, then probes onnxruntime + llama-cpp-python CUDA support. On success it surfaces `SUPERGRAPH_GPU=1` so the existing compute_profile gate flips automatically. Failure is structured (`status.error`) and falls back to CPU silently.

For per-component control, pass the explicit kwargs (CPU stays the default):

```python
SuperGraph(path="./brain", gpu_layers=-1, reranker_gpu_layers=-1)
BonsaiIngestor(model_path=..., n_gpu_layers=-1)
```

## ✨ Pro mode

`pip install 'supergraphdb[pro]'` bundles ingest + vision + audio + embedders-extra + gpu plus huggingface-hub / tokenizers / onnxruntime. Pair it with a one-time calibration to get spec-driven validation and a calibrated Bonsai ingestor without writing the device-detection / sizing / fallback glue yourself.

```bash
pip install 'supergraphdb[pro]'
supergraph pro setup        # download every component, probe each on this host
supergraph pro status       # inspect host + spec + resolved knobs
```

```python
from supergraph import SuperGraph

gs = SuperGraph(path="./brain", profile="pro")

# Resolver caught every shortfall up-front (extras missing, calibration
# stale, RAM/VRAM short). If we got here, the spec runs.
print(gs.pro_resolved.n_ctx, gs.pro_resolved.bonsai_n_gpu_layers)

ing = gs.create_bonsai()                       # n_ctx / n_batch / n_gpu_layers
ing.ingest("Maria joined OpenAI.", msg_id="m1")  # all wired from calibration
```

Defaults match the measured-best LoCoMo configuration as of this release: `jina-v5-small` embedder, `jina-v3` reranker, `bonsai-tq1_0-lite` ingest, `tinybert` NER. Customize via a `ProSpec(...)` instance. Strict by default: missing extras / missing calibration / unfit host raise `ProExtraNotInstalled` / `ProCalibrationMissing` / `ProUnsupportedHostError`. Pass `pro_strict=False` to log + continue. Linux x86_64 + NVIDIA CUDA 12 only in v1.

Full guide: [Pro mode](website/docs/guides/pro-mode.md).

## 🐳 Docker

Two images. Both expose the playground HTTP API on `:7200` and persist the database on a `/data` volume.

```bash
# CPU image (~660 MB) - fits 90% of deploys
docker compose up -d                              # builds + starts on :7200

# Pro GPU image (~5.2 GB slim, includes pre-pulled Bonsai/jina/tinybert weights)
docker compose --profile pro up -d                # requires nvidia-container-toolkit
docker compose --profile pro run --rm supergraph-pro supergraph pro setup   # one-time calibration
```

The Pro image is a single-stage `python:3.12-slim` + pip-delivered nvidia CUDA wheels (cu12 runtime + cuDNN 9 + cuBLAS) + GPU-built `llama-cpp-python`. All install + symbol-strip + execution-provider prune happens in one RUN to avoid the layer-overhead bug where a later `strip` adds duplicate copies on top of originals. Set `--build-arg SKIP_MODEL_PREFETCH=1` for a ~3.5 GB image that downloads models on first use instead.

The entrypoint auto-generates an auth token on first boot and writes it to `/data/.auth_token` (printed to logs). Use it as `Authorization: Bearer <token>` on every `/api/*` call. To pin a fixed token set `SUPERGRAPH_AUTH_TOKEN` in the environment; to disable auth on a private network set `SUPERGRAPH_ALLOW_UNAUTH_BIND=1`.

```bash
TOKEN=$(docker exec supergraph cat /data/.auth_token)
curl -s -X POST http://127.0.0.1:7200/api/execute \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"query":"COUNT NODES"}'
# {"kind":"count","data":0,"count":0,...}
```

Resource limits in `docker-compose.yml` cap each container at 8 CPUs / 16 GB RAM. Adjust via `deploy.resources.limits` per host.

## 🔌 MCP server (agentic memory)

supergraph ships a Model Context Protocol server that exposes the store as agent-callable tools (Claude Desktop, Cursor, any MCP-aware client). No playground HTTP server required - the server holds an in-process `SuperGraph()` and translates typed tool calls into DSL.

```bash
pip install 'supergraphdb[mcp]'   # adds the mcp Python SDK
supergraph-mcp                  # stdio server, ready for Claude Desktop
```

Tools exposed: `gs_remember(text)`, `gs_remember_batch(texts)`, `sg_search(query)` (3-signal fusion), `gs_recall(node_id)`, `gs_lexical(query)`, `gs_similar(text)`, `gs_traverse(from_id)`, `sg_answer(query)` (RAG via Bonsai in Pro mode), `gs_count_nodes()`, plus `sg_execute(dsl)` as a raw-DSL escape hatch. Errors return structured `{"error": "..."}` instead of crashing the transport.

Claude Desktop config snippet (see `tools/mcp/claude_desktop_config.example.json`):

```json
{
  "mcpServers": {
    "supergraph": {
      "command": "supergraph-mcp",
      "env": { "SUPERGRAPH_DB_PATH": "/path/to/supergraph-agent.db" }
    }
  }
}
```

For Pro mode (Bonsai LLM + Jina + NER), add `"SUPERGRAPH_PROFILE": "pro"`. Set `SUPERGRAPH_URL=http://host:7200` to forward calls to a shared remote playground instead of running in-process.

## 🎯 Scope

- Embedded, one writer per path. For multi-tenant, wrap in your own service.
- No SQL, no Cypher, no distributed cluster. Graph ops exist because agent memory is a graph.
- Fusion weights are hand-tuned. Reranking is opt-in, off by default.
- Bonsai NL→DSL ingest is opt-in via `BonsaiIngestor(...)`. Core install never auto-loads an LLM.
- GPU offload is opt-in via `gpu.setup()` or explicit `n_gpu_layers=...`. No silent device acquisition.

## 🛠️  Development

```bash
git clone https://github.com/orkait/supergraph.git
cd supergraph
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ingest,vision,embedders-extra,playground]"
pytest
```

Docs site under `website/` (Docusaurus). Run locally:

```bash
cd website && bun install && bun run start
```

## 📄 License

AGPL-3.0. See [LICENSE](LICENSE).
