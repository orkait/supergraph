"""Cloud-backed NL->DSL ingestor.

Same @-verb shorthand + whole-turn DSL synthesis as the local BonsaiIngestor
(reused via the synthesis shim), but generation runs through litellm with a
free-tier-first multi-provider chain. Adds a streaming progress API.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator

from supergraph.llm_runner import LLMRunner
from supergraph.ingest.llm.resolve import build_provider_chain, DEFAULT_FREE_FIRST_CHAIN
from supergraph.ingest.llm import synthesis as S

# cloud.py is at src/supergraph/ingest/llm/cloud.py; the skill prompt is at
# src/supergraph/bonsai_dsl_prompt.txt -> two parents up from `llm/`.
_DEFAULT_SKILL = Path(__file__).resolve().parents[2] / "bonsai_dsl_prompt.txt"


class CloudIngestor:
    """NL -> DSL ingestion through cloud LLM providers.

    Reuses Bonsai's prompt + parse + synthesize pipeline; only the LLM
    transport differs. Tracks cross-message belief state like Bonsai so the
    model reuses fact ids across calls.
    """

    def __init__(
        self,
        gs: Any | None = None,
        *,
        models: list[str] | None = None,
        free_first: bool = True,
        aliases: dict[str, str] | None = None,
        skill_path: str | Path | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        retries: int = 3,
        timeout_s: int = 90,
    ) -> None:
        self._gs = gs
        chain = build_provider_chain(
            models or DEFAULT_FREE_FIRST_CHAIN,
            free_first=free_first,
            aliases=aliases,
        )
        if not chain:
            raise S.IngestError(
                "no cloud providers resolved; set at least one provider API key "
                "(GROQ_API_KEY / CEREBRAS_API_KEY / CLOUDFLARE_API_KEY+ACCOUNT_ID / "
                "GOOGLE_AISTUDIO_API_KEY / OPENROUTER_API_KEY)"
            )
        self._runner = LLMRunner(chain, retries=retries, timeout_s=timeout_s)
        self._skill_path = Path(skill_path) if skill_path else _DEFAULT_SKILL
        self._system_prompt = self._skill_path.read_text()
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._facts: dict[str, S.FactState] = {}

    def _messages(self, text: str) -> list[dict]:
        facts_block = S.render_known_facts_block(self._facts)
        parts = [p for p in (facts_block, text) if p]
        user = "\n\n".join(parts) if parts else text
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user},
        ]

    def _synthesize(self, cleaned, *, msg_id, session_id, role, text, dry_run):
        """Parse @-verbs -> synthesize DSL -> parse-validate. No execution.

        Returns (valid_lines, rejected). Synthesis uses the live gs (unless
        dry_run) so entity resolution dedupes against the real store.
        """
        turn = S.parse_verb_output(cleaned)
        synthesized = S.synthesize_dsl(
            turn, msg_id=msg_id, session_id=session_id, role=role, text=text,
            gs=None if dry_run else self._gs,
        )
        from supergraph.dsl.parser import parse as dsl_parse
        valid: list[str] = []
        rejected: list[tuple[str, str]] = []
        for ln in synthesized:
            try:
                dsl_parse(ln)
                valid.append(ln)
            except Exception as err:
                rejected.append((ln, f"parse error: {err}"))
        return valid, rejected

    def _execute_iter(self, valid: list[str], *, dry_run: bool) -> Iterator[dict]:
        """Execute each valid line, yielding one event per line.

        Goes through gs.execute so the single-writer / queued contract holds.
        Belief-state scrape is the caller's job (needs the full executed set).
        """
        if dry_run:
            for ln in valid:
                yield {"statement": ln, "status": "dry_run"}
            return
        for ln in valid:
            try:
                self._gs.execute(ln)
                yield {"statement": ln, "status": "ok"}
            except Exception as err:
                yield {"statement": ln, "status": "rejected", "error": str(err)}

    def ingest(
        self,
        text: str,
        *,
        msg_id: str,
        session_id: str = "default",
        role: str = "user",
        dry_run: bool = False,
    ) -> S.IngestResult:
        """Generate (batch), synthesize, execute. Returns IngestResult."""
        if not text or not text.strip():
            raise S.IngestEmpty("input text is empty or whitespace-only")
        if not dry_run and self._gs is None:
            raise ValueError("ingest requires a SuperGraph (pass gs=...) or dry_run=True")
        t0 = time.perf_counter()
        raw = self._runner.complete_messages(
            self._messages(text), max_tokens=self._max_tokens, temperature=self._temperature,
        )
        cleaned = S.strip_think(raw)
        if not cleaned:
            raise S.IngestEmpty(f"LLM returned empty or <think>-only output. raw={raw!r}")

        valid, rejected = self._synthesize(
            cleaned, msg_id=msg_id, session_id=session_id, role=role, text=text, dry_run=dry_run,
        )
        events = list(self._execute_iter(valid, dry_run=dry_run))
        executed_lines = [e["statement"] for e in events if e["status"] == "ok"]
        rejected += [(e["statement"], e["error"]) for e in events if e["status"] == "rejected"]
        if not dry_run:
            S.scrape_belief_updates(executed_lines, self._facts)
        return S.IngestResult(
            statements=list(valid),
            executed=len(executed_lines),
            rejected=rejected,
            entities_new=[],
            beliefs_changed=[],
            duration_ms=int((time.perf_counter() - t0) * 1000),
            raw_output=raw,
            dry_run=dry_run,
        )

    def ingest_stream(
        self,
        text: str,
        *,
        msg_id: str,
        session_id: str = "default",
        role: str = "user",
    ) -> Iterator[dict]:
        """Stream generation, then synthesize + execute whole-turn.

        Yields progress events:
          {"phase": "generating", "delta": str}    one per token chunk
          {"phase": "synthesizing"}                 once, before execution
          {"phase": "executing", "statement", "status": "ok"|"rejected", "error"?}
          {"phase": "done", "status": "ok"|"empty", "executed"?, "rejected"?}

        The model emits @-verb shorthand, so DSL synthesis needs the whole
        turn; execution is whole-turn, not per-token. Streaming gives live
        generation progress + live per-statement execution feedback.
        """
        if not text or not text.strip():
            raise S.IngestEmpty("input text is empty or whitespace-only")
        if self._gs is None:
            raise ValueError("ingest_stream requires a SuperGraph (pass gs=...)")
        buf: list[str] = []
        for delta in self._runner.stream_messages(
            self._messages(text), max_tokens=self._max_tokens, temperature=self._temperature,
        ):
            buf.append(delta)
            yield {"phase": "generating", "delta": delta}

        cleaned = S.strip_think("".join(buf))
        if not cleaned:
            yield {"phase": "done", "status": "empty"}
            return

        yield {"phase": "synthesizing"}
        valid, rejected = self._synthesize(
            cleaned, msg_id=msg_id, session_id=session_id, role=role, text=text, dry_run=False,
        )
        for rej_line, reason in rejected:
            yield {"phase": "executing", "statement": rej_line, "status": "rejected", "error": reason}

        executed_lines: list[str] = []
        for ev in self._execute_iter(valid, dry_run=False):
            if ev["status"] == "ok":
                executed_lines.append(ev["statement"])
            yield {"phase": "executing", **ev}

        S.scrape_belief_updates(executed_lines, self._facts)
        yield {
            "phase": "done",
            "status": "ok",
            "executed": len(executed_lines),
            "rejected": len(valid) - len(executed_lines) + len(rejected),
        }
