"""LLM-driven skill-guided ingest adapter.

Pattern B from the architecture discussion: instead of deterministic parse +
NER + CREATE NODE (the baseline `supergraph_.py` adapter does that), this
adapter hands each session to an LLM along with the `supergraph-dsl` skill
and asks the LLM to emit DSL statements. Each emitted line is parsed
through Lark; valid statements execute, invalid ones get counted and
dropped.

Why this shape:
- supergraph's DSL is small (~7 constructs for ingest) -> LLM can learn it
- skill text is tight (~19 KB) -> fits cleanly in context
- parser roundtrip guarantees no silent corruption
- Python-side escape helpers never run; LLM must escape its own strings
  per the skill's rules

Bench: same query side as `supergraph_.py`; only `ingest()` differs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from supergraph.dsl.parser import parse as dsl_parse

from .base import Session, TimedOperation
from ..transport.llm_client import llm_call
from .supergraph_ import SuperGraphAdapter


_SKILL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "tools" / "skills" / "supergraph-dsl" / "SKILL.md"

# Regexes to scrape ASSERT + RETRACT statements from emitted DSL so we can
# track running belief state across sessions within one conversation. These
# match the canonical shape the skill tells the LLM to emit; anything odd
# just misses the scrape and stays uninjected - harmless.
_ASSERT_RE = re.compile(
    r'^ASSERT\s+"([^"\\]+(?:\\.[^"\\]*)*)"\s+(.*)$',
    re.IGNORECASE,
)
_RETRACT_RE = re.compile(
    r'^RETRACT\s+"([^"\\]+(?:\\.[^"\\]*)*)"(?:\s+REASON\s+"([^"\\]*(?:\\.[^"\\]*)*)")?\s*$',
    re.IGNORECASE,
)
_KV_VALUE_RE = re.compile(r'value\s*=\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
_KV_KIND_RE = re.compile(r'kind\s*=\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
_KV_CONFIDENCE_RE = re.compile(r'CONFIDENCE\s+([0-9.]+)')
_KV_SOURCE_RE = re.compile(r'SOURCE\s+"([^"\\]*(?:\\.[^"\\]*)*)"')
_KV_EVENT_AT_RE = re.compile(r'EVENT_AT\s+"([^"\\]*(?:\\.[^"\\]*)*)"')


@dataclass
class _FactState:
    """One live belief tracked across sessions."""
    fact_id: str
    kind: str = ""
    value: str = ""
    confidence: float = 1.0
    source: str = ""
    event_at: str = ""
    retracted: bool = False
    retract_reason: str = ""


@dataclass
class _IngestStats:
    emitted: int = 0
    executed: int = 0
    parse_failed: int = 0
    exec_failed: int = 0
    llm_empty: int = 0
    sessions: int = 0
    last_parse_errors: list[tuple[str, str]] = None  # type: ignore
    last_exec_errors: list[tuple[str, str]] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.last_parse_errors is None:
            self.last_parse_errors = []
        if self.last_exec_errors is None:
            self.last_exec_errors = []

    def as_dict(self) -> dict[str, Any]:
        return {
            "emitted": self.emitted,
            "executed": self.executed,
            "parse_failed": self.parse_failed,
            "exec_failed": self.exec_failed,
            "llm_empty": self.llm_empty,
            "sessions": self.sessions,
            "accept_rate": round(self.executed / max(self.emitted, 1), 3),
        }


def _render_known_facts_block(facts: dict[str, _FactState], max_facts: int = 120) -> str:
    """Format live (non-retracted) facts so the LLM can see prior belief state.

    Retracted facts are skipped so the LLM cannot re-assert something that was
    already superseded. If the set grows beyond max_facts, keep the most
    recent by insertion order (dict preserves it in Python 3.7+).
    """
    alive = [f for f in facts.values() if not f.retracted]
    if not alive:
        return ""
    alive = alive[-max_facts:]
    lines = ["--- KNOWN FACTS FROM PRIOR SESSIONS ---",
             "# Do not re-assert any of these. If a new message contradicts one,",
             "# emit RETRACT before the superseding ASSERT.",
             ""]
    for f in alive:
        bits = [f'[{f.fact_id}]']
        if f.kind:
            bits.append(f'kind="{f.kind}"')
        bits.append(f'value="{f.value}"')
        bits.append(f'confidence={f.confidence:.2f}')
        if f.source:
            bits.append(f'source="{f.source}"')
        if f.event_at:
            bits.append(f'event_at="{f.event_at}"')
        lines.append(" ".join(bits))
    return "\n".join(lines)


def _scrape_belief_updates(
    executed_lines: list[str],
    facts: dict[str, _FactState],
) -> None:
    """Walk successfully-executed statements and update running fact state.

    Only ASSERT / RETRACT matter. Other writes ignored.
    """
    for line in executed_lines:
        m = _ASSERT_RE.match(line)
        if m:
            fact_id = m.group(1)
            rest = m.group(2)
            st = facts.get(fact_id) or _FactState(fact_id=fact_id)
            km = _KV_KIND_RE.search(rest)
            if km:
                st.kind = km.group(1)
            vm = _KV_VALUE_RE.search(rest)
            if vm:
                st.value = vm.group(1)
            cm = _KV_CONFIDENCE_RE.search(rest)
            if cm:
                try:
                    st.confidence = float(cm.group(1))
                except ValueError:
                    pass
            sm = _KV_SOURCE_RE.search(rest)
            if sm:
                st.source = sm.group(1)
            em = _KV_EVENT_AT_RE.search(rest)
            if em:
                st.event_at = em.group(1)
            st.retracted = False
            st.retract_reason = ""
            facts[fact_id] = st
            continue
        m = _RETRACT_RE.match(line)
        if m:
            fact_id = m.group(1)
            reason = m.group(2) or ""
            st = facts.get(fact_id) or _FactState(fact_id=fact_id)
            st.retracted = True
            st.retract_reason = reason
            facts[fact_id] = st


def _render_session_prompt(
    session: Session,
    skill: str,
    known_facts: dict[str, _FactState] | None = None,
) -> str:
    """Build the LLM prompt: skill + known-facts block + session + instructions."""
    date = session.metadata.get("date", "unknown-date")
    turns = []
    for i, msg in enumerate(session.messages):
        turns.append(f"[{i}] [{date}] {msg.role}: {msg.content}")
    turns_block = "\n".join(turns)

    facts_block = ""
    if known_facts:
        rendered = _render_known_facts_block(known_facts)
        if rendered:
            facts_block = rendered + "\n\n"

    instructions = f"""
--- YOUR TASK ---

You have been given the `supergraph-dsl` skill above. You are now ingesting one
conversation session into supergraph. Your job: emit supergraph DSL statements,
one per line, that store this session as durable memory.

Session id: {session.session_id}
Date: {date}
Messages: {len(session.messages)}

Rules:
- Emit plain DSL text only. No Python. No code fences. No prose, no numbering.
- One statement per line. No blank lines inside a statement.
- Escape `"` as `\\"` inside every string literal.
- Schema is already registered (kinds: session, message, entity). Do NOT emit
  `SYS REGISTER` statements.
- Create one `session` node with EVENT_AT set to the date:
  CREATE NODE "sess:{session.session_id}" kind = "session" session_id = "{session.session_id}" EVENT_AT "{date}"
- For each message, CREATE NODE with:
    kind = "message"
    session = "{session.session_id}"
    role = "user"|"assistant"
    position = <zero-based index>
    content = "<the raw message text, quote-escaped>"
    EVENT_AT "{date}"
  Use ids of the form `{session.session_id}:msg<i>`. Do NOT use a DOCUMENT
  clause - use the `content` field only. The schema is registered with
  EMBED content, which auto-embeds the content field on CREATE.
- Add `CREATE EDGE "sess:<id>" -> "<session_id>:msg<i>" kind = "has_message"`
  for every message.
- Between adjacent messages, add `CREATE EDGE "<session_id>:msg<i>" -> "<session_id>:msg<i+1>" kind = "next"`.
- Extract prominent named entities (people, places, concrete objects) from
  each message. For each unique entity in a message:
    UPSERT NODE "ent:<slug>" kind = "entity" name = "<display-name>"
    CREATE EDGE "<session_id>:msg<i>" -> "ent:<slug>" kind = "mentions"
  where <slug> is lowercase + underscores, stripped of punctuation.
- Dedupe entity mentions per message (G8 in the skill). Do not emit two
  `mentions` edges from the same message to the same entity.
- If a message asserts a durable fact (preferences, allergies, pets,
  relationships), additionally emit:
    ASSERT "fact:<slug>" kind = "fact" value = "<value>" CONFIDENCE <0.5-1.0> SOURCE "<session_id>:msg<i>" EVENT_AT "{date}"
  with confidence reflecting how explicit the statement is.
- If a later message in this session contradicts an earlier fact asserted in
  this same session, emit `RETRACT "fact:<slug>" REASON "<why>"` after the
  superseding `ASSERT`.

--- RAW TURNS ---

{turns_block}

--- EMIT DSL NOW ---
""".strip()

    return f"{skill}\n\n{facts_block}{instructions}"


_STATEMENT_BATCH_LIMIT = 400
_FENCE_RE = re.compile(r"^```(?:\w+)?\s*$", re.MULTILINE)
_PROSE_PREFIX_RE = re.compile(r"^(?:#|//|--|\*|\d+[\.)]\s|- |\* )")


def _iter_dsl_lines(raw: str) -> list[str]:
    """Strip code fences + prose, return candidate DSL lines."""
    cleaned = _FENCE_RE.sub("", raw)
    out: list[str] = []
    for line in cleaned.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _PROSE_PREFIX_RE.match(line):
            continue
        out.append(line)
    return out


class SuperGraphSkillAdapter(SuperGraphAdapter):
    """LLM-driven ingest. Query path inherited from SuperGraphAdapter."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.stats = _IngestStats()
        self._skill: str | None = None
        self._max_tokens: int = int(self.config.get("skill_max_tokens", 6000))
        self._retry_on_empty: int = int(self.config.get("skill_retry_on_empty", 1))
        self._carry_facts: bool = bool(self.config.get("skill_carry_facts", True))
        self._max_known_facts: int = int(self.config.get("skill_max_known_facts", 120))
        dump = self.config.get("skill_dump_raw_dir")
        self._dump_dir: Path | None = Path(dump) if dump else None
        if self._dump_dir is not None:
            self._dump_dir.mkdir(parents=True, exist_ok=True)
        self._last_raw: str | None = None
        self._known_facts: dict[str, _FactState] = {}
        self.name = f"{self.name}-skill-llm"

    @property
    def last_raw_output(self) -> str | None:
        """Raw LLM output from the most recent ingest call. None before first call."""
        return self._last_raw

    def _load_skill(self) -> str:
        if self._skill is not None:
            return self._skill
        if not _SKILL_PATH.exists():
            raise FileNotFoundError(f"supergraph-dsl skill not found at {_SKILL_PATH}")
        self._skill = _SKILL_PATH.read_text(encoding="utf-8")
        return self._skill

    def ingest(self, session: Session) -> float:
        if self._gs is None:
            raise RuntimeError("reset() must be called first")
        if not session.messages:
            return 0.0

        skill = self._load_skill()
        known = self._known_facts if self._carry_facts else None
        prompt = _render_session_prompt(session, skill, known_facts=known)

        with TimedOperation() as t:
            raw = llm_call(prompt, max_tokens=self._max_tokens, _retries=self._retry_on_empty)
            self._last_raw = raw
            if self._dump_dir is not None:
                safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", session.session_id)
                (self._dump_dir / f"{safe_id}.dsl").write_text(raw or "", encoding="utf-8")
            if not raw:
                self.stats.llm_empty += 1
                self.stats.sessions += 1
                return t.elapsed_ms

            lines = _iter_dsl_lines(raw)
            self.stats.emitted += len(lines)

            executed_lines: list[str] = []
            with self._gs.deferred_embeddings(batch_size=self._embed_batch_size):
                for line in lines[:_STATEMENT_BATCH_LIMIT]:
                    try:
                        dsl_parse(line)
                    except Exception as e:
                        self.stats.parse_failed += 1
                        if len(self.stats.last_parse_errors) < 5:
                            self.stats.last_parse_errors.append((line[:120], str(e)[:120]))
                        continue
                    try:
                        self._gs.execute(line)
                        self.stats.executed += 1
                        executed_lines.append(line)
                    except Exception as e:
                        self.stats.exec_failed += 1
                        if len(self.stats.last_exec_errors) < 5:
                            self.stats.last_exec_errors.append((line[:120], str(e)[:120]))
                        continue

            if self._carry_facts:
                _scrape_belief_updates(executed_lines, self._known_facts)
                # cap memory: drop oldest retracted, then oldest live, beyond limit
                if len(self._known_facts) > self._max_known_facts * 2:
                    # Evict retracted first
                    for fid in list(self._known_facts.keys()):
                        if len(self._known_facts) <= self._max_known_facts * 2:
                            break
                        if self._known_facts[fid].retracted:
                            del self._known_facts[fid]
                    # Then oldest live
                    while len(self._known_facts) > self._max_known_facts * 2:
                        self._known_facts.pop(next(iter(self._known_facts)))

        self.stats.sessions += 1
        return t.elapsed_ms

    def ingest_done(self, record_metadata: dict[str, Any] | None = None) -> None:
        """Emit ingest stats at end of record for the benchmark runner to log."""
        if record_metadata is not None:
            record_metadata.setdefault("ingest_stats", {}).update(self.stats.as_dict())

    def reset(self) -> None:
        super().reset()
        self.stats = _IngestStats()
        self._known_facts = {}
        # Match the baseline schema exactly: `content:string` REQUIRED +
        # `EMBED content`. This is load-bearing for A/B parity - parent's
        # query strategies read `n.get("content")`, which is empty if the
        # LLM uses `DOCUMENT "..."` (blob-only path). The prompt instructs
        # the LLM to emit `content = "..."` as a typed field so retrieval
        # finds real text.
        # Parent skipped `entity` kind when entity_extraction=False. Add it
        # here - the LLM is expected to emit UPSERT NODE for entities.
        try:
            self._gs.execute('SYS REGISTER NODE KIND "entity" REQUIRED name:string')
        except Exception:
            pass  # already registered
