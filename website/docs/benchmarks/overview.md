---
title: Overview
sidebar_position: 1
---

# Benchmarks

Benchmark results, methodology, and reproduction instructions.

:::info Metric callout
supergraph ships retrieval numbers, not end-to-end QA accuracy. *Retrieval accuracy* below means "did the gold-answer-bearing passage land in the retrieved top-K". End-to-end QA with an LLM reader is a strict superset.
:::

## LongMemEval-S

Retrieval-only benchmark. 500 records, each with ~53 sessions (~500 messages). Per-record isolated evaluation: reset -> ingest haystack -> query -> check if answer session ID in retrieved results.

### Latest results (Kaggle, 2026-04-19)

Jina v5 Small 1024d, Kaggle T4 GPU. Public kernel (full logs + reproducible in-browser): [kaggle.com/code/superkaiii/supergraph-jina-v5-small](https://www.kaggle.com/code/superkaiii/supergraph-jina-v5-small).

| Category | n | Retrieval accuracy |
|---|---|---|
| knowledge-update | 78 | 100.0% |
| multi-session | 133 | 98.5% |
| single-session-assistant | 56 | 100.0% |
| single-session-user | 70 | 98.6% |
| temporal-reasoning | 133 | 94.7% |
| single-session-preference | 30 | 83.3% |
| **Overall** | **500** | **97.0%** |

Latency: query p50 46 ms / p95 76 ms, ingest p50 1035 ms / p95 1070 ms. Memory delta +283 MB across 23,867 ingest ops. 5 h 20 m wall. No LLM judge, zero API calls.

### How to reproduce

**Docker:**

```bash
python -m benchmarks.framework.docker_runner \
  --embedder onnx \
  --embedder-model-dir /path/to/jina-nano \
  --data /path/to/longmemeval \
  --gpu
```

**Kaggle:**

```bash
# Update benchmarks/kaggle/supergraph_jina_500.py with your HF token
kaggle kernels push -p benchmarks/kaggle
```

## LoCoMo

End-to-end QA benchmark. 10 conversations, ~200 QAs each, 1986 total. Ingest all sessions for a conversation once, query all QAs against same state. Token-level F1 with Porter stemming (official protocol from snap-research/locomo).

### 50Q random sample

conv-26, MiniMax M2.7 nitro, Jina v5 Small 1024d:

| Category | n | F1 |
|---|---|---|
| open-domain | 10 | 0.452 |
| multi-hop | 10 | 0.418 |
| adversarial | 10 | 0.500 |
| single-hop | 10 | 0.224 |
| temporal | 10 | 0.189 |
| **Overall** | **50** | **0.357** |

For context: GPT-3.5-turbo-16k with full conversation context scores 0.378 on LoCoMo. supergraph hits comparable quality using only retrieved passages (no full context), with a smaller reader LLM.

### Retrieval recall at K (no LLM)

| K | Recall |
|---|---|
| top-5 | 60% |
| top-10 | 80% |
| top-20 | 84% |
| top-50 | 96% |

Measured on 50 validated questions where the keyword exists in the ingested data.

### How to reproduce

```bash
# Download locomo10.json from https://huggingface.co/datasets/Percena/locomo-mc10
# Place at /tmp/locomo/raw/locomo10.json

# Install jina-v5-small
SUPERGRAPH_MODEL_CACHE_DIR=/tmp/gs_models python -c "
from supergraph.registry.installer import install_embedder, set_cache_dir
set_cache_dir('/tmp/gs_models')
install_embedder('jina-v5-small-retrieval')
"

# Run full benchmark (requires LLM - set OPENROUTER_API_KEY/OLLAMA_API_KEY in /.env;
# QA_MODEL_PRIORITY lives in src/supergraph/llm_runner.py)
python -m benchmarks.framework.runners.locomo \
  --data-path /tmp/locomo \
  --embedder installed:jina-v5-small-retrieval \
  --k 10

# Run direct evidence recall test (no LLM needed)
python -m benchmarks.framework.runners.ratchet_recall
```

For a walkthrough of ingestion and querying on a single LoCoMo conversation, see [First memory](../guides/first-memory).

## Micro-latency (single operation)

Median latency over 30 iters, model2vec 256-dim embeddings, 16-core CPU @ 2-thread BLAS cap. Reproduce: `python benchmarks/micro_latency.py`. Last measured 2026-04-19.

| Operation | In-memory | On-disk | Notes |
|---|---|---|---|
| Point lookup `NODE "id"` | **5 us** | 11 us | hash -> slot |
| Filtered scan `NODES WHERE ... LIMIT 10` | **14 us** | 51 us | typed column filter |
| Semantic search `SIMILAR TO "..." LIMIT 10` | **87 us** | 175 us | usearch HNSW ANN |
| Graph traversal `RECALL DEPTH 3` | ~1 ms | ~1 ms | spreading activation |
| Hybrid retrieval `REMEMBER LIMIT 10` | ~6 ms | ~50 ms | 4-signal fusion (scales with candidate set) |
| `ASSERT` | 11 us | 4 ms | disk path pays WAL sync per call |
| Memory per node | ~1.6 KB | ~1.6 KB | ~80 bytes typed columns + ~1 KB vector + overhead |

Disk numbers at **100k nodes**, in-memory at **10k nodes** (disk WAL sync dominates at small N, ANN tree depth dominates at large N). REMEMBER scales with the number of candidates the ANN + FTS leg return - realistic workloads have far fewer than 100 matches and the fused pipeline drops to single-digit ms.

## Methodology

- LongMemEval scoring: substring match of any gold answer in retrieved text, OR answer session ID match in retrieved node metadata
- LoCoMo scoring: official token-level F1 with Porter stemming, Counter-based multiset overlap, per-category handling (multi-hop splits sub-answers, adversarial checks "no information available")
- The LLM reader (MiniMax M2.7) generates a short answer from retrieved context. F1 is computed between this answer and the gold answer. No LLM judge involved.
- MemMachine and other SOTA systems report LLM-judge accuracy (binary correct/wrong), which inflates ~1.5-2x compared to token F1. Numbers are not directly comparable.
- Retrieval recall uses keyword matching (does the longest distinctive word from the gold answer appear in the retrieved text?) which is conservative - the LLM might extract the answer even without an exact keyword match.

## BEAM

supergraph includes a benchmark-side BEAM answer-generation runner at `benchmarks/framework/runners/beam.py`. Keeps supergraph core untouched and emits BEAM-compatible answer JSON so BEAM's own evaluator can score it.

### Workflow

1. Prepare a local BEAM checkout or dataset directory
2. Run supergraph answer generation against BEAM chats
3. Run BEAM's official evaluator on the produced answer files

### Example

```bash
# Generate answers for BEAM 100K chats 1..2
uv run python3 -m benchmarks.framework.runners.beam \
  --beam-root /tmp/BEAM \
  --chat-size 100K \
  --start-index 1 \
  --end-index 3 \
  --retrieval-method pair_chunk \
  --embedder installed:jina-v5-small-retrieval \
  --embedder-cache-dir /tmp/gs_models \
  --reader-model-name gpt-4.1-mini \
  --reader-model-url https://api.openai.com/v1 \
  --reader-model-api-key "$OPENAI_API_KEY" \
  --result-file-name supergraph_beam_answers.json \
  --k 5

# Evaluate with BEAM's official scorer
cd /tmp/BEAM
python -m src.evaluation.run_evaluation \
  --input_directory results/100K \
  --chat_size 100K \
  --start_index 0 \
  --end_index 2 \
  --max_workers 4 \
  --allowed_result_files supergraph_beam_answers.json
```

Supported chunk modes: `pair_chunk`, `turn_chunk`. The runner builds one supergraph per BEAM chat, ingests once, and answers all probing questions against that live state. This is answer-generation support, not a separate custom evaluator.

## Run all three benchmarks from one CLI

```bash
python -m benchmarks.framework.cli run --dataset longmemeval --data-path ./data --variant s
python -m benchmarks.framework.cli run --dataset locomo --data-path ./data
python -m benchmarks.framework.cli run --dataset beam --data-path /tmp/BEAM --variant 16k --end-index 10
```
