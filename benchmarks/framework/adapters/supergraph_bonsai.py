"""Bonsai-driven ingest + recall adapter for LoCoMo / LongMemEval.

Replaces:
  - Ingest path: the deterministic NER+CREATE NODE pipeline in supergraph_.py
    (and the remote-LLM path in supergraph_skill.py). Instead, every user
    turn goes through a local Ternary-Bonsai 4B TQ1_0 via BonsaiIngestor,
    which emits @-verb lines that Python synthesizes into DSL.
  - Query DSL emission: the hard-coded REMEMBER/RECALL dispatch in
    supergraph_.py. Instead, every question goes through the same
    BonsaiIngestor (dry_run=True) which emits @REMEMBER/@SIMILAR/@LEXICAL/
    @ANSWER/@RECALL lines; the first retrieval-shaped statement is executed
    against the SuperGraph.

What's NOT replaced:
  - Downstream answer-synthesis LLM (Phase 2 in locomo.py) still runs; F1
    compares its output vs gold. This adapter only owns the NL->DSL path.
  - Retrieval ranking (REMEMBER fusion, RECALL graph walk) still happens
    inside supergraph; the adapter just picks which DSL form runs.

Single ingestor instance is reused across conversations (loading the 4B
GGUF is 8-20s). Between conversations we swap the `gs` reference and
clear cross-message fact state so each conversation starts fresh.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import QueryContext, QueryResult, Session, TimedOperation
from .supergraph_ import SuperGraphAdapter, _escape


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_MODEL = _REPO_ROOT / "models" / "Ternary-Bonsai-4B-GGUF" / "Ternary-Bonsai-4B-TQ1_0.gguf"

_RETRIEVAL_PREFIXES = (
    "REMEMBER ", "SIMILAR TO ", "LEXICAL SEARCH ",
    "RECALL FROM ", "TRAVERSE FROM ", "ANCESTORS OF ",
    "DESCENDANTS OF ", "SUBGRAPH FROM ",
    "PATH FROM ", "SHORTEST PATH ", "COMMON NEIGHBORS ",
    "ANSWER ",
)


@dataclass
class _BonsaiStats:
    ingest_turns: int = 0
    ingest_skipped: int = 0
    query_turns: int = 0
    query_fallbacks: int = 0  # Bonsai emitted no retrieval op -> fell back to vanilla REMEMBER
    parse_errors: int = 0
    exec_errors: int = 0


class SuperGraphBonsaiAdapter(SuperGraphAdapter):
    """Bonsai-driven NL interpretation for ingestion and recall."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.stats = _BonsaiStats()
        self._bonsai: Any | None = None  # BonsaiIngestor - lazy-loaded
        self._model_path = Path(self.config.get("bonsai_model_path", _DEFAULT_MODEL))
        self._prompt_path_conf = self.config.get("bonsai_prompt_path")
        self._n_gpu_layers = int(self.config.get("bonsai_n_gpu_layers", 0))
        self._n_ctx = self.config.get("bonsai_n_ctx")  # None -> auto
        self._max_output_tokens = int(self.config.get("bonsai_max_output_tokens", 160))
        self._kv_cache_path = self.config.get("bonsai_kv_cache_path")
        self.name = f"{self.name}-bonsai"

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------

    def _ensure_bonsai(self) -> Any:
        """Lazy-load BonsaiIngestor once per adapter lifetime."""
        if self._bonsai is not None:
            # reuse: swap supergraph reference + clear fact state
            self._bonsai._gs = self._gs
            self._bonsai.reset_facts()
            return self._bonsai
        from supergraph.bonsai_ingestor import (
            BonsaiIngestor, _DEFAULT_LITE_PROMPT_PATH,
        )
        skill_path = (
            Path(self._prompt_path_conf) if self._prompt_path_conf
            else _DEFAULT_LITE_PROMPT_PATH
        )
        kw: dict[str, Any] = {
            "model_path": self._model_path,
            "gs": self._gs,
            "skill_path": skill_path,
            "max_output_tokens": self._max_output_tokens,
            "n_gpu_layers": self._n_gpu_layers,
        }
        if self._n_ctx is not None:
            kw["n_ctx"] = int(self._n_ctx)
        if self._kv_cache_path:
            kw["kv_cache_path"] = Path(self._kv_cache_path)
        self._bonsai = BonsaiIngestor(**kw)
        self._bonsai.warmup()
        return self._bonsai

    def reset(self) -> None:
        """Fresh SuperGraph + reuse warm Bonsai (just rebind gs + clear facts)."""
        super().reset()
        self._ensure_bonsai()

    # ---------------------------------------------------------------
    # Ingest: one user turn per Bonsai call
    # ---------------------------------------------------------------

    def ingest(self, session: Session) -> float:
        if self._gs is None:
            raise RuntimeError("reset() must be called first")
        if not session.messages:
            return 0.0
        ing = self._ensure_bonsai()
        ing._gs = self._gs  # ensure live supergraph is the target

        with TimedOperation() as t:
            for i, msg in enumerate(session.messages):
                # LoCoMo stores role as the speaker name (e.g. "Caroline");
                # LongMemEval uses "user"/"assistant". Accept any non-empty
                # message as ingestable - every conversational turn carries
                # memory-relevant content.
                if not msg.content or not msg.content.strip():
                    continue
                msg_id = f"m:{session.session_id}:{i}"
                try:
                    ing.ingest(
                        msg.content,
                        msg_id=msg_id,
                        session_id=session.session_id,
                        role=msg.role or "user",
                    )
                    self.stats.ingest_turns += 1
                except Exception:
                    self.stats.ingest_skipped += 1
        return t.elapsed_ms

    # ---------------------------------------------------------------
    # Query: Bonsai parses the NL question -> first retrieval DSL runs
    # ---------------------------------------------------------------

    def query_with_context(self, ctx: QueryContext, k: int = 5) -> QueryResult:
        if self._gs is None:
            raise RuntimeError("reset() must be called first")
        ing = self._ensure_bonsai()
        self.stats.query_turns += 1

        with TimedOperation() as t:
            retrieval_stmt = self._emit_retrieval_stmt(ing, ctx.question, k)
            rows = self._exec_retrieval(retrieval_stmt, ctx.question, k)
            memories = self._texts(rows)[:k]
        return QueryResult(
            retrieved_memories=memories,
            elapsed_ms=t.elapsed_ms,
            raw=rows,
        )

    def _emit_retrieval_stmt(self, ing: Any, question: str, k: int) -> str:
        """Ask Bonsai to parse the question; find the first retrieval statement."""
        try:
            r = ing.ingest(
                question,
                msg_id=f"q:{hash(question) & 0xffffffff:x}",
                session_id="__query__",
                role="user",
                dry_run=True,
            )
        except Exception:
            self.stats.parse_errors += 1
            return self._vanilla_remember(question, k)

        for stmt in r.statements:
            up = stmt.upper()
            if any(up.startswith(p) for p in _RETRIEVAL_PREFIXES):
                # Bonsai default LIMIT 10; respect the runner's k when smaller
                return self._apply_limit(stmt, k)

        # No retrieval op emitted -> fall back
        self.stats.query_fallbacks += 1
        return self._vanilla_remember(question, k)

    @staticmethod
    def _apply_limit(stmt: str, k: int) -> str:
        """Rewrite trailing LIMIT N to the requested k if N > k."""
        m = re.search(r"\bLIMIT\s+(\d+)\b", stmt, re.IGNORECASE)
        if m:
            existing = int(m.group(1))
            if existing > k:
                return stmt[: m.start()] + f"LIMIT {k}" + stmt[m.end():]
        return stmt

    @staticmethod
    def _vanilla_remember(question: str, k: int) -> str:
        q = _escape(question)
        return f'REMEMBER "{q}" LIMIT {k} WHERE kind = "message"'

    def _exec_retrieval(self, stmt: str, question: str, k: int) -> list[dict]:
        try:
            result = self._gs.execute(stmt)
            data = result.data
            return data if isinstance(data, list) else []
        except Exception:
            self.stats.exec_errors += 1
            # Last-ditch fallback
            try:
                result = self._gs.execute(self._vanilla_remember(question, k))
                return result.data if isinstance(result.data, list) else []
            except Exception:
                return []

    # ---------------------------------------------------------------
    # Observability
    # ---------------------------------------------------------------

    def ingest_done(self, record_metadata: dict[str, Any] | None = None) -> None:
        super().ingest_done(record_metadata=record_metadata)
        if record_metadata is not None:
            record_metadata.setdefault("bonsai_stats", {}).update({
                "ingest_turns": self.stats.ingest_turns,
                "ingest_skipped": self.stats.ingest_skipped,
                "query_turns": self.stats.query_turns,
                "query_fallbacks": self.stats.query_fallbacks,
                "parse_errors": self.stats.parse_errors,
                "exec_errors": self.stats.exec_errors,
            })

    def close(self) -> None:
        super().close()
        # Let Python GC the BonsaiIngestor -> Llama instance.
        self._bonsai = None
