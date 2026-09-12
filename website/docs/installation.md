---
title: Installation
sidebar_position: 2
---

# Installation

## Core

```bash
pip install supergraph
```

Covers the agentic DB contract out of the box: `REMEMBER` / `RECALL` (model2vec embedder), `SYS CRON` (croniter), `VAULT SYNC` (pyyaml), plus the numpy / scipy / usearch / lark / msgspec / psutil / threadpoolctl foundation. No torch, no PDF parser, no HTTP server.

## Optional extras

```bash
# PDF / DOCX / HTML ingestion (+200 MB)
pip install 'supergraph[ingest]'

# Local VLM sidecar for scanned PDFs or image captioning (+80 MB wheel, ~1.5 GB weights on first use)
pip install 'supergraph[vision]'

# Speech-to-text (wav/mp3/flac/m4a)
pip install 'supergraph[audio]'

# GPU acceleration for NER (Linux x86_64, CUDA 12)
pip install 'supergraph[gpu]'

# Pro profile: spec-driven, calibrated agentic-memory deployment
# (bundles ingest + vision + audio + embedders-extra + gpu plus
# huggingface-hub + tokenizers + onnxruntime). See "Pro mode" guide.
pip install 'supergraph[pro]'

# Everything heavy
pip install 'supergraph[ingest,vision,playground]'
```

## All extras

| Extra | What it adds |
|---|---|
| `ingest` | markitdown + pymupdf + pymupdf4llm (PDF/DOCX/HTML to markdown) |
| `ingest-pro` | docling (heavier PDF w/ tables + OCR; ~1 GB via torch). For CPU-only: `pip install 'supergraph[ingest-pro]' --extra-index-url https://download.pytorch.org/whl/cpu` |
| `vision` | llama-cpp-python[server] + huggingface-hub (local VLM sidecar, SmolVLM2-2.2B Q4_K_M ~1.5 GB on first use) |
| `audio` | faster-whisper (in-process speech-to-text; tiny/base models ~40-150 MB on first use) |
| `embedders-extra` | fastembed + llama-cpp-python (alternate embedder backends; model2vec is default and core) |
| `playground` | fastapi + uvicorn (local web UI) |
| `gpu` | onnxruntime-gpu only (bring your own CUDA 12 + cuDNN 9) |
| `pro` | All-in-one agentic memory: ingest + vision + audio + embedders-extra + gpu + tokenizers + onnxruntime + huggingface-hub. Backed by `supergraph pro setup` calibration. |
| `dev` | pytest + pytest-benchmark + pytest-xdist + pytest-timeout |

## Verify

```python
from supergraph import SuperGraph
g = SuperGraph(path=":memory:")
g.execute('CREATE NODE "hello" kind = "memory" DOCUMENT "world"')
print(g.execute('REMEMBER "world" LIMIT 1').data)
g.close()
```

## Python support

Python 3.10, 3.11, 3.12, 3.13. CI runs all four on every PR.
