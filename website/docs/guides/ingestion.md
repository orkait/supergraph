---
title: Ingesting files
sidebar_position: 2
---

# Ingesting files

`INGEST` is a first-class DSL verb. It dispatches to a tiered, modality-aware pipeline and lands into the same three storage engines as direct writes.

```sql
INGEST "report.pdf" AS "doc:q3" KIND "report"
SYS CONNECT    -- auto-wire similar chunks across documents
```

## Format coverage

Core install handles `txt / md / csv / json / html`. File formats needing extra machinery:

| Extra | Formats | How |
|---|---|---|
| `[ingest]` | `pdf / docx / xlsx / pptx / html` | markitdown to pymupdf4llm (PDFs) |
| `[ingest-pro]` | same + `tex / adoc / tiff / bmp` + richer PDFs | docling (~1 GB, pulls torch) |
| `[vision]` | `png / jpg / webp` + scanned PDF fallback | local llama.cpp sidecar + SmolVLM2-2.2B (~1.5 GB on first call); auto-starts on first `INGEST ... USING VISION` |
| `[audio]` | `wav / mp3 / flac / m4a / opus / webm` | in-process faster-whisper (~150 MB on first call); timestamp-tagged chunks |

See [Installation](../installation) for extra details.

## Examples

```sql
INGEST "scan.pdf"                                     -- whatever tier applies
INGEST "chart.png" USING VISION "smolvlm2-2.2b"
INGEST "interview.mp3"                                -- needs [audio]
```

## Bring your own VLM

Point at any OpenAI-compatible vision endpoint (Ollama, vLLM, OpenAI) via `SUPERGRAPH_VISION_URL`. See `supergraph vision {serve|stop|status|logs|models}` for the local sidecar.

## DOCUMENT caveat

Plain `CREATE NODE "id" kind = "X" topic = "..."` without a `DOCUMENT` clause stores typed columns only - `REMEMBER` and `LEXICAL` return zero for that node. Use `DOCUMENT "text"` whenever the node's content is what you want to retrieve on.

## Persistence

Everything persists to `./brain/` as SQLite. Reopen with the same path and all memories are back.
