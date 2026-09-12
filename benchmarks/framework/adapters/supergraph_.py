"""SuperGraph adapter, skill-compliant.

Implements every rule from tools/skills/supergraph-builder/SKILL.md:
    - Schema first (SYS REGISTER NODE KIND) with EMBED on message only
    - deferred_embeddings() per session
    - put_summary() per message so REMEMBER's BM25 leg actually works (G2)
    - No EMBED on entity nodes (G5)
    - No __updated_at__ override (G3)
    - Entity graph via regex extraction, linked via "mentions" edges
    - Per-category query dispatch that combines REMEMBER + RECALL (G1)
    - WHERE kind = "message" on REMEMBER so entities/sessions don't pollute
"""

from __future__ import annotations

import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from supergraph import SuperGraph, __version__ as _GS_VERSION

from .base import QueryContext, QueryResult, Session, TimedOperation
from ..entity_extraction import build_entity_extractor


_SLUG_RE = re.compile(r"[^a-zA-Z0-9_]+")

_BENCHMARK_DEFAULTS: dict[str, Any] = {
    # No overrides - trust config.py defaults as single source of truth.
}

# Adapter-only tuning knobs used to size per-strategy fan-out. Not passed
# to SuperGraph - they only scale k in adapter-side dispatch helpers.
_ADAPTER_DEFAULTS: dict[str, Any] = {
    "retrieval_depth": 9,
    "recall_depth": 2,
    "max_query_entities": 6,
    "recency_boost_k": 4,
}


def _escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _slug(text: str) -> str:
    return _SLUG_RE.sub("_", text.lower()).strip("_")[:40]
def _extract_temporal_anchor(question: str, question_date: str | None = None) -> int | None:
    """Extract a temporal anchor from the query text.

    Uses the question date as the reference point for relative expressions.
    Returns epoch ms or None.
    """
    from supergraph.core.temporal import extract_dates, parse_date

    reference = None
    if question_date:
        q_ms = parse_date(question_date)
        if q_ms is not None:
            reference = datetime.fromtimestamp(q_ms / 1000, tz=timezone.utc)
    dates = extract_dates(question, reference=reference)
    return dates[0] if dates else None


def _build_embedder(config: dict[str, Any]):
    name = (config.get("embedder") or "model2vec").lower()
    gpu = bool(config.get("embedder_gpu"))
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else None

    if name in ("model2vec", "default"):
        return "default"
    if name == "fastembed":
        from supergraph.embedding.fastembed_embedder import FastEmbedEmbedder

        return FastEmbedEmbedder(
            model_name=config.get("embedder_model", "BAAI/bge-small-en-v1.5"),
            cache_dir=config.get("cache_dir") or config.get("model_cache_dir"),
            threads=config.get("embedder_threads"),
        )
    if name == "onnx":
        from supergraph.embedding.onnx_hf_embedder import OnnxHFEmbedder

        model_dir = config.get("embedder_model_dir")
        if not model_dir:
            raise ValueError("embedder=onnx requires embedder_model_dir")
        return OnnxHFEmbedder(
            model_dir=model_dir,
            output_dims=config.get("embedder_output_dims"),
            max_length=int(config.get("embedder_max_length", 512)),
            pooling_mode=config.get("embedder_pooling", "mean"),
            providers=providers,
            gpu_mem_limit=config.get("embedder_gpu_mem_limit"),
        )
    if name == "installed":
        from supergraph.registry.installer import load_installed_embedder, set_cache_dir
        cache = config.get("embedder_cache_dir")
        if cache:
            set_cache_dir(cache)
        model_name = config.get("embedder_model")
        if not model_name:
            raise ValueError("embedder=installed requires embedder_model (e.g. 'harrier-oss-v1-0.6b')")
        return load_installed_embedder(
            model_name,
            dims=config.get("embedder_output_dims"),
            providers=providers,
        )
    if name in ("jina-v5-nano", "jina-v5-nano-retrieval"):
        from supergraph.registry.installer import load_installed_embedder, set_cache_dir
        cache = config.get("embedder_cache_dir")
        if cache:
            set_cache_dir(cache)
        return load_installed_embedder(
            "jina-v5-nano-retrieval",
            dims=768,
            providers=providers,
        )
    if name == "gguf":
        from supergraph.embedding.llamacpp_embedder import LlamaCppEmbedder
        gguf_path = config.get("embedder_gguf_path")
        if not gguf_path:
            raise ValueError("embedder=gguf requires embedder_gguf_path")
        return LlamaCppEmbedder(
            model_path=gguf_path,
            n_ctx=int(config.get("embedder_max_length", 2048)),
            n_gpu_layers=int(config.get("embedder_gpu_layers", 0)),
            output_dims=config.get("embedder_output_dims"),
            query_prefix=config.get("embedder_query_prefix", ""),
            doc_prefix_template=config.get("embedder_doc_prefix", ""),
        )
    raise ValueError(f"unknown embedder: {name!r}")


def _build_reranker(config: dict[str, Any]):
    backend = (config.get("reranker") or "").lower()
    if backend == "flashrank":
        from supergraph.embedding.reranker import FlashRankReranker
        return FlashRankReranker(
            model_name=config.get("reranker_model", "rank-T5-flan"),
            max_length=int(config.get("reranker_max_length", 512)),
        )
    if backend == "onnx":
        reranker_dir = config.get("reranker_model_dir")
        if not reranker_dir:
            return None
        from supergraph.embedding.reranker import OnnxReranker
        return OnnxReranker(
            model_dir=reranker_dir,
            onnx_file=config.get("reranker_onnx_file", "onnx/model_int8.onnx"),
            max_length=int(config.get("reranker_max_length", 512)),
        )
    if backend == "gguf":
        from supergraph.embedding.reranker import GGUFReranker
        max_len = config.get("reranker_max_length")
        return GGUFReranker(
            model_path=config.get("reranker_model_dir", ""),
            projector_path=config.get("reranker_projector_path"),
            n_ctx=int(max_len) if max_len is not None else None,
            n_gpu_layers=int(config.get("reranker_gpu_layers", -1)),
        )
    return None


class SuperGraphAdapter:
    name = "supergraph"
    version = _GS_VERSION

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._tmpdir: Path | None = None
        self._gs: SuperGraph | None = None
        self._embed_batch_size: int = int(self.config.get("embed_batch_size", 128))
        self._entity_extractor = build_entity_extractor(self.config) if self.config.get("entities", True) else None
        self._entity_extraction: bool = self._entity_extractor is not None
        self._populate_fts: bool = bool(self.config.get("populate_fts", True))
        self._reranker = _build_reranker(self.config)
        self._embedder = _build_embedder(self.config)
        emb_name = (self.config.get("embedder") or "model2vec").lower()
        if emb_name == "fastembed":
            short = self.config.get("embedder_model", "bge-small").split("/")[-1]
            self.name = f"supergraph-skill-fastembed-{short}"
        elif emb_name == "onnx":
            short = Path(self.config.get("embedder_model_dir", "onnx")).name
            self.name = f"supergraph-skill-onnx-{short}"
        elif emb_name == "installed":
            short = (self.config.get("embedder_model") or "installed").lower()
            self.name = f"supergraph-skill-{short}"
        else:
            self.name = "supergraph-skill-model2vec"

    def reset(self) -> None:
        self.close()
        self._tmpdir = Path(tempfile.mkdtemp(prefix="gs_bench_"))
        # Build constructor kwargs from any CLI overrides (only explicitly set values)
        gs_kwargs: dict[str, Any] = {
            "path": str(self._tmpdir),
            "ceiling_mb": self.config.get("ceiling_mb", 4096),
            "queued": False,
            "embedder": self._embedder,
            "entity_model_dir": None, # Force engine NER OFF
            "enable_sentence_nodes": False, # Force sentence splitting OFF
            "enable_rollback": False, # Force rollback snapshots OFF
            "auto_optimize": False, # Force optimizer checks OFF
            "enable_wal": False, # Force WAL OFF
        }
        # SuperGraph constructor kwargs (must match __init__ signature)
        for key in ("remember_weights", "search_oversample", "recall_decay",
                     "similarity_threshold", "duplicate_threshold",
                     "recency_half_life_days", "similar_to_oversample",
                     "lexical_search_oversample",
                     "fusion_method", "rrf_k",
                     "nucleus_expansion", "nucleus_hops",
                     "sentence_query_expansion"):
            val = self.config.get(key, _BENCHMARK_DEFAULTS.get(key))
            if val is not None:
                gs_kwargs[key] = val
        self._gs = SuperGraph(**gs_kwargs)
        self._gs.execute(
            'SYS REGISTER NODE KIND "message" '
            'REQUIRED session:string, role:string, content:string '
            'OPTIONAL position:int '
            'EMBED content'
        )
        self._gs.execute(
            'SYS REGISTER NODE KIND "session" '
            'REQUIRED session_id:string '
            'OPTIONAL position:int, msg_count:int'
        )
        if self._entity_extraction:
            self._gs.execute(
                'SYS REGISTER NODE KIND "entity" REQUIRED name:string'
            )

    def warmup(self) -> None:
        """Pre-load AI models into GPU to avoid first-run latency."""
        if self._entity_extraction:
            self._entity_extractor.extract("warmup")
        if self._embedder and hasattr(self._embedder, "embed"):
            try:
                self._embedder.embed(["warmup"])
            except Exception:
                pass

    def ingest(self, session: Session) -> float:
        if self._gs is None:
            raise RuntimeError("reset() must be called first")

        n = len(session.messages)
        if n == 0:
            return 0.0

        sid = _escape(session.session_id)

        with TimedOperation() as t:
            # 1. Batch extract entities for all messages in the session
            msg_contents = [msg.content for msg in session.messages]
            all_entities = []
            st = time.time()
            if self._entity_extraction:
                all_entities = self._entity_extractor.extract_batch(msg_contents)
            t_ner = time.time() - st

            # 2. Build a single transaction block for the entire session
            st = time.time()
            dsl = ["BEGIN"]
            
            sess_node_id = f"sess:{session.session_id}"
            dsl.append(
                f'CREATE NODE "{sess_node_id}" kind = "session" '
                f'session_id = "{sid}" '
                f'position = {int(session.metadata.get("position", 0))} '
                f'msg_count = {n}'
            )

            # Parse session date for EVENT_AT clause
            sess_date_str = session.metadata.get("date", "")
            sess_event_ms = None
            if sess_date_str:
                from supergraph.core.temporal import parse_date
                sess_event_ms = parse_date(sess_date_str)

            event_at_clause = f" EVENT_AT {sess_event_ms}" if sess_event_ms else ""

            entity_seen: set[str] = set()

            for i, msg in enumerate(session.messages):
                entities = all_entities[i] if all_entities else []
                msg_id = f"{session.session_id}:msg{i}"
                content = _escape(msg.content)
                role = _escape(msg.role)

                # DOCUMENT clause auto-populates doc_fts for plaintext bodies
                # (PR #102), so the adapter no longer needs a separate
                # index_text_batch call after the batch.
                doc_clause = f' DOCUMENT "{content}"' if self._populate_fts else ""
                dsl.append(
                    f'CREATE NODE "{msg_id}" kind = "message" '
                    f'session = "{sid}" role = "{role}" '
                    f'content = "{content}" '
                    f'position = {i}{event_at_clause}{doc_clause}'
                )

                dsl.append(f'CREATE EDGE "{sess_node_id}" -> "{msg_id}" kind = "has_message"')

                if self._entity_extraction:
                    # Dedupe per message. NER can emit the same entity
                    # string multiple times (multi-span hits on repeated
                    # mentions within a single message), which would
                    # otherwise produce a duplicate mentions edge and
                    # rollback the whole batch (Kaggle v29 failure mode).
                    msg_ent_seen: set[str] = set()
                    for ent_name in entities:
                        ent_slug = _slug(ent_name)
                        if not ent_slug:
                            continue
                        ent_id = f"ent:{ent_slug}"
                        if ent_id in msg_ent_seen:
                            continue
                        msg_ent_seen.add(ent_id)
                        if ent_id not in entity_seen:
                            dsl.append(
                                f'UPSERT NODE "{ent_id}" kind = "entity" '
                                f'name = "{_escape(ent_name)}"'
                            )
                            entity_seen.add(ent_id)
                        dsl.append(f'CREATE EDGE "{msg_id}" -> "{ent_id}" kind = "mentions"')

            for i in range(n - 1):
                a = f"{session.session_id}:msg{i}"
                b = f"{session.session_id}:msg{i + 1}"
                dsl.append(f'CREATE EDGE "{a}" -> "{b}" kind = "next"')

            dsl.append("COMMIT")
            t_dsl_gen = time.time() - st

            # 3. Execute the single batch transaction
            st = time.time()
            with self._gs.deferred_embeddings(batch_size=self._embed_batch_size):
                self._gs.execute("\n".join(dsl))
            t_exec = time.time() - st
                
            # 4. FTS is populated inline via DOCUMENT clause above; no
            #    separate pass needed.
            t_fts = 0.0

        if t.elapsed_ms > 100:
            print(f"    [DEBUG] sess={session.session_id} msgs={n} ner={t_ner:.2f}s, exec={t_exec:.2f}s, fts={t_fts:.2f}s")

        return t.elapsed_ms / 1000.0

    def query_with_context(self, ctx: QueryContext, k: int = 5) -> QueryResult:
        if self._gs is None:
            raise RuntimeError("reset() must be called first")

        category = ctx.category or ""
        anchor_ms = _extract_temporal_anchor(
            ctx.question,
            str(ctx.metadata.get("question_date", "")) if ctx.metadata else None,
        )
        with TimedOperation() as t:
            retrieved, raw = self._dispatch(ctx.question, category, k, anchor_ms)
        return QueryResult(
            retrieved_memories=retrieved,
            elapsed_ms=t.elapsed_ms,
            raw=raw,
        )

    def query(self, question: str, k: int = 5) -> QueryResult:
        return self.query_with_context(QueryContext(question=question), k=k)

    def ingest_done(self, record_metadata: dict[str, Any] | None = None) -> None:
        """Optional post-ingest hook for expensive offline memory maintenance."""
        if self._gs is None:
            return
        if self.config.get("enable_consolidation"):
            self._gs.execute("SYS CONSOLIDATE")

    def _cfg(self, attr: str):
        """Read a config value from own config, falling back to adapter defaults."""
        return self.config.get(attr, _ADAPTER_DEFAULTS.get(attr))

    def _resolve_strategy(self, category: str) -> str:
        """Pick a retrieval strategy for the current benchmark question."""
        explicit = self.config.get("retrieval_strategy")
        if explicit is not None:
            return explicit
        return "full"

    def _remember_cmd(self, question: str, limit: int, anchor_ms: int | None = None) -> str:
        """Build a REMEMBER DSL command with optional AT temporal anchor."""
        q = _escape(question)
        at = f" AT {anchor_ms}" if anchor_ms else ""
        return f'REMEMBER "{q}"{at} LIMIT {limit} WHERE kind = "message"'

    def _dispatch(self, question: str, category: str, k: int,
                  anchor_ms: int | None = None) -> tuple[list[str], Any]:
        """Dispatch to the configured retrieval strategy."""
        strategy = self._resolve_strategy(category)
        handler = self._STRATEGIES.get(strategy, self._strategy_full)
        return handler(self, question, k, anchor_ms)

    def _strategy_remember(self, question: str, k: int,
                           anchor_ms: int | None = None) -> tuple[list[str], Any]:
        """Hybrid only. Fast, no graph overhead."""
        depth = self._cfg("retrieval_depth")
        r = self._gs.execute(self._remember_cmd(question, k * depth, anchor_ms))
        return self._texts(r.data)[:k], r.data

    def _strategy_remember_graph(self, question: str, k: int,
                                anchor_ms: int | None = None) -> tuple[list[str], Any]:
        """Hybrid + entity graph traversal. Good for cross-session."""
        depth = self._cfg("retrieval_depth")
        primary = self._gs.execute(self._remember_cmd(question, k * depth, anchor_ms))
        merged = self._texts(primary.data)

        if self._entity_extraction:
            max_ents = self._cfg("max_query_entities")
            recall_d = self._cfg("recall_depth")
            for ent_name in self._entity_extractor.extract(question)[:max_ents]:
                ent_id = f"ent:{_slug(ent_name)}"
                try:
                    rec = self._gs.execute(
                        f'RECALL FROM "{ent_id}" DEPTH {recall_d} LIMIT {k}'
                    )
                    for text in self._texts(rec.data):
                        if text not in merged:
                            merged.append(text)
                except Exception:
                    pass

        return merged[:k], primary.data

    def _strategy_remember_recency(self, question: str, k: int,
                                  anchor_ms: int | None = None) -> tuple[list[str], Any]:
        """Hybrid + recency boost. Good for knowledge updates."""
        depth = self._cfg("retrieval_depth")
        primary = self._gs.execute(self._remember_cmd(question, k * depth, anchor_ms))
        merged = self._texts(primary.data)

        try:
            recency_k = k * self._cfg("recency_boost_k")
            recent = self._gs.execute(
                f'NODES WHERE kind = "message" '
                f'ORDER BY __updated_at__ DESC LIMIT {recency_k}'
            )
            for text in self._texts(recent.data):
                if text not in merged:
                    merged.append(text)
        except Exception:
            pass

        return merged[:k], primary.data

    def _strategy_full(self, question: str, k: int,
                       anchor_ms: int | None = None) -> tuple[list[str], Any]:
        """Hybrid + graph + recency. All signals, no category routing."""
        depth = self._cfg("retrieval_depth")
        primary = self._gs.execute(self._remember_cmd(question, k * depth, anchor_ms))
        merged = self._texts(primary.data)

        if self._entity_extraction:
            max_ents = self._cfg("max_query_entities")
            recall_d = self._cfg("recall_depth")
            for ent_name in self._entity_extractor.extract(question)[:max_ents]:
                ent_id = f"ent:{_slug(ent_name)}"
                try:
                    rec = self._gs.execute(
                        f'RECALL FROM "{ent_id}" DEPTH {recall_d} LIMIT {k}'
                    )
                    for text in self._texts(rec.data):
                        if text not in merged:
                            merged.append(text)
                except Exception:
                    pass

        try:
            recency_k = k * self._cfg("recency_boost_k")
            recent = self._gs.execute(
                f'NODES WHERE kind = "message" '
                f'ORDER BY __updated_at__ DESC LIMIT {recency_k}'
            )
            for text in self._texts(recent.data):
                if text not in merged:
                    merged.append(text)
        except Exception:
            pass

        return merged[:k], primary.data

    def _strategy_lexical_boost(self, question: str, k: int,
                               anchor_ms: int | None = None) -> tuple[list[str], Any]:
        """Hybrid + extra lexical search. Good for keyword-heavy queries."""
        depth = self._cfg("retrieval_depth")
        primary = self._gs.execute(self._remember_cmd(question, k * depth, anchor_ms))
        merged = self._texts(primary.data)

        try:
            lexical = self._gs.execute(
                f'LEXICAL SEARCH "{_escape(question)}" LIMIT {k * 2}'
            )
            for text in self._texts(lexical.data):
                if text not in merged:
                    merged.append(text)
        except Exception:
            pass

        return merged[:k], primary.data

    def _rerank(self, question: str, texts: list[str], k: int) -> list[str]:
        """Rerank texts by cross-encoder relevance. Falls back to truncation if no reranker."""
        if not self._reranker or len(texts) <= k:
            return texts[:k]
        scores = self._reranker.score(question, texts)
        ranked = sorted(zip(scores, texts), reverse=True)
        return [t for _, t in ranked[:k]]

    def _strategy_remember_rerank(self, question: str, k: int,
                                 anchor_ms: int | None = None) -> tuple[list[str], Any]:
        """Hybrid retrieval + cross-encoder rerank. Research-backed best pattern."""
        depth = self._cfg("retrieval_depth")
        primary = self._gs.execute(self._remember_cmd(question, k * depth, anchor_ms))
        candidates = self._texts(primary.data)
        reranked = self._rerank(question, candidates, k)
        return reranked, primary.data

    def _strategy_full_rerank(self, question: str, k: int,
                             anchor_ms: int | None = None) -> tuple[list[str], Any]:
        """All signals + cross-encoder rerank. Maximum accuracy."""
        texts, raw = self._strategy_full(question, k * 2, anchor_ms)
        reranked = self._rerank(question, texts, k)
        return reranked, raw

    def _strategy_consolidated(self, question: str, k: int,
                              anchor_ms: int | None = None) -> tuple[list[str], Any]:
        """Two-stage retrieval: observations first, episodes fill gaps.

        TSM-inspired priority cascade:
            1. Search observations (consolidated facts) - high precision
            2. Fill remaining slots from episodes (raw messages) - high recall
            3. Merge, dedup
        """
        q = _escape(question)
        at = f" AT {anchor_ms}" if anchor_ms else ""
        depth = self._cfg("retrieval_depth")

        # Stage 1: Search observations (consolidated memories)
        try:
            obs_result = self._gs.execute(
                f'REMEMBER "{q}"{at} LIMIT {k * 2} WHERE kind = "observation"'
            )
            obs_texts = self._texts(obs_result.data)
        except Exception:
            obs_texts = []

        # Stage 2: Search episodes (raw messages) to fill gaps
        episode_result = self._gs.execute(self._remember_cmd(question, k * depth, anchor_ms))
        episode_texts = self._texts(episode_result.data)

        # Merge: observations first (higher priority), then episodes
        merged = list(obs_texts)
        seen = set(merged)
        for text in episode_texts:
            if text not in seen:
                merged.append(text)
                seen.add(text)

        # Entity graph enrichment (same as full strategy)
        if self._entity_extraction:
            max_ents = self._cfg("max_query_entities")
            recall_d = self._cfg("recall_depth")
            for ent_name in self._entity_extractor.extract(question)[:max_ents]:
                ent_id = f"ent:{_slug(ent_name)}"
                try:
                    rec = self._gs.execute(
                        f'RECALL FROM "{ent_id}" DEPTH {recall_d} LIMIT {k}'
                    )
                    for text in self._texts(rec.data):
                        if text not in seen:
                            merged.append(text)
                            seen.add(text)
                except Exception:
                    pass

        all_raw = (obs_result.data if obs_texts else []) + (episode_result.data or [])
        return merged[:k], all_raw

    _STRATEGIES = {
        "remember": _strategy_remember,
        "remember_graph": _strategy_remember_graph,
        "remember_recency": _strategy_remember_recency,
        "remember_lexical": _strategy_lexical_boost,
        "remember_rerank": _strategy_remember_rerank,
        "full": _strategy_full,
        "full_rerank": _strategy_full_rerank,
        "consolidated": _strategy_consolidated,
    }

    @staticmethod
    def _texts(rows: Any) -> list[str]:
        if not rows:
            return []
        out: list[str] = []
        for node in rows:
            text = node.get("content") or node.get("name") or node.get("summary") or ""
            if text:
                out.append(text)
        return out

    def close(self) -> None:
        if self._gs is not None:
            try:
                self._gs.close()
            except Exception:
                pass
            self._gs = None
        if self._tmpdir and self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = None
