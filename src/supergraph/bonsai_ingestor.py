"""Natural-language to DSL ingestor backed by a local llama.cpp GGUF.

Target model: Ternary-Bonsai 4B TQ1_0 (CPU + RAM only, offline).

Download the model once before first use (models/ is gitignored):

    mkdir -p models/Ternary-Bonsai-4B-TQ1_0 && curl -L -o \\
      models/Ternary-Bonsai-4B-TQ1_0/Ternary-Bonsai-4B-TQ1_0.gguf \\
      https://huggingface.co/superkaiii/Ternary-Bonsai-4B-GGUF/resolve/main/Ternary-Bonsai-4B-TQ1_0.gguf

Publication pipeline: benchmarks/kaggle/pack_ternary_bonsai/ converts
prism-ml/Ternary-Bonsai-4B-unpacked (FP16) to TQ1_0 via a Kaggle kernel and
publishes the result to superkaiii/Ternary-Bonsai-4B-GGUF on HF.

The public surface is `BonsaiIngestor`. Every call:
  1. Loads the skill prompt once, fingerprints it, pins that fingerprint into
     the system prompt so KV cache stays coherent across calls. When the skill
     file on disk changes, the next call naturally forces a warm re-process
     because the system-prompt prefix now differs.
  2. Serializes access to the `Llama` instance with a lock - llama-cpp-python
     is NOT thread-safe; concurrent calls corrupt the KV cache / segfault.
  3. Checks the combined prompt + output budget against `n_ctx` and resets the
     cache if the request would force a head-trim (which silently evicts the
     skill prefix and causes quality collapse).
  4. Treats empty / <think>-only outputs as ingestion errors, not as silent
     no-ops (the behaviour-analysis audit called this out - silent failures
     hid real bugs).
  5. Dedupes UPSERT NODE by id before handing the DSL to the parser, so a
     BatchRollback from a duplicated entity never takes out an otherwise
     valid batch.
  6. Emits one structured log line per ingest with counts for every
     category: statements emitted, parsed, rejected, executed, duration.
  7. Supports `dry_run=True` to generate the DSL and return it without
     touching the SuperGraph - used for testing, previewing, and building
     training data.

It also tracks cross-message belief state. After every successful ingest, the
executed ASSERT / RETRACT lines are scraped into a running `fact_id -> FactState`
dict. The next ingest injects the live (non-retracted) facts into the USER
message (not the system prompt - see below), so the model sees prior fact ids
and reuses them when updating the same concept. Without this, the model coins
a new fact_id per message and the graph ends up with multiple beliefs for the
same underlying concept.

The known-facts block goes in the USER message on purpose: the system prompt
stays byte-identical across calls, which keeps llama.cpp's prefix-match KV
cache warm. If we appended facts to the system prompt, every call would be a
cold one.

Not handled here:
  - Streaming. Generation is blocking. Callers can run it in a thread.
  - Multi-user / multi-tenant. Use one BonsaiIngestor per user.
  - Model swap. Create a new BonsaiIngestor; don't mutate this one's paths.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------

class BonsaiError(Exception):
    """Base class for ingestor errors."""


class IngestEmpty(BonsaiError):
    """LLM returned empty or <think>-only output. No DSL to execute."""


class IngestOverflow(BonsaiError):
    """Requested prompt + output would exceed n_ctx even after a cache reset."""


# --------------------------------------------------------------------
# Result
# --------------------------------------------------------------------

@dataclass
class FactState:
    """One live belief tracked across messages within this ingestor.

    Updated by `_scrape_belief_updates` after every successful ingest and
    rendered back into the next ingest's user message by
    `_render_known_facts_block`, so the model reuses the same fact_id for the
    same concept instead of coining a new one each call.
    """

    fact_id: str
    kind: str = ""
    value: str = ""
    confidence: float = 1.0
    source: str = ""
    retracted: bool = False
    retract_reason: str = ""


@dataclass
class IngestResult:
    """Everything an ingest call produced, for inspection and tracing."""

    statements: list[str] = field(default_factory=list)
    executed: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)
    entities_new: list[str] = field(default_factory=list)
    beliefs_changed: list[tuple[str, str]] = field(default_factory=list)
    duration_ms: int = 0
    raw_output: str = ""
    skill_fingerprint: str = ""
    dry_run: bool = False


# --------------------------------------------------------------------
# Post-processing helpers
# --------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*$|^```\s*$")
_UPSERT_RE = re.compile(r'^\s*UPSERT\s+NODE\s+"([^"\\]+(?:\\.[^"\\]*)*)"', re.IGNORECASE)
_ASSERT_LINE_RE = re.compile(
    r'^\s*ASSERT\s+"([^"\\]+(?:\\.[^"\\]*)*)"\s+(.*)$',
    re.IGNORECASE,
)
_ASSERT_RE = re.compile(r'^\s*ASSERT\s+"([^"\\]+(?:\\.[^"\\]*)*)"', re.IGNORECASE)
_RETRACT_RE = re.compile(
    r'^\s*RETRACT\s+"([^"\\]+(?:\\.[^"\\]*)*)"(?:\s+REASON\s+"([^"\\]*(?:\\.[^"\\]*)*)")?\s*$',
    re.IGNORECASE,
)
_CREATE_NODE_RE = re.compile(r'^\s*CREATE\s+NODE\s+"([^"\\]+(?:\\.[^"\\]*)*)"', re.IGNORECASE)
_ENT_FROM_ID_RE = re.compile(r'"(ent:[^"\\]+)"')
_KV_VALUE_RE = re.compile(r'value\s*=\s*"([^"\\]*(?:\\.[^"\\]*)*)"', re.IGNORECASE)
_KV_KIND_RE = re.compile(r'kind\s*=\s*"([^"\\]*(?:\\.[^"\\]*)*)"', re.IGNORECASE)
_KV_CONFIDENCE_RE = re.compile(r'CONFIDENCE\s+([0-9.]+)', re.IGNORECASE)
_KV_SOURCE_RE = re.compile(r'SOURCE\s+"([^"\\]*(?:\\.[^"\\]*)*)"', re.IGNORECASE)


def _strip_think(raw: str) -> str:
    """Remove reasoning-model `<think>...</think>` wrappers."""
    return _THINK_RE.sub("", raw).strip()


def _split_lines(cleaned: str) -> list[str]:
    """One statement per line. Drop markdown fences and blanks."""
    out: list[str] = []
    for raw_ln in cleaned.splitlines():
        ln = raw_ln.strip()
        if not ln or _FENCE_RE.match(ln):
            continue
        out.append(ln)
    return out


def _dedupe_upserts(stmts: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Keep the first UPSERT per entity id. Drop later duplicates.

    Duplicate UPSERT NODE with the same id inside a BEGIN/COMMIT batch makes
    the rollback snapshot logic trip over itself - entire batch errors out.
    Pre-filter here so the transaction never sees the duplicate.
    """
    seen: set[str] = set()
    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    for ln in stmts:
        m = _UPSERT_RE.match(ln)
        if m:
            eid = m.group(1)
            if eid in seen:
                dropped.append((ln, f"duplicate UPSERT of {eid!r}"))
                continue
            seen.add(eid)
        kept.append(ln)
    return kept, dropped


def _scrape_belief_updates(
    executed_lines: list[str],
    facts: dict[str, FactState],
) -> None:
    """Walk successfully-executed DSL lines and update running fact state.

    Only ASSERT / RETRACT contribute. Other writes ignored. Mutates `facts`
    in place.
    """
    for line in executed_lines:
        m = _ASSERT_LINE_RE.match(line)
        if m:
            fact_id = m.group(1)
            rest = m.group(2)
            st = facts.get(fact_id) or FactState(fact_id=fact_id)
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
            st.retracted = False
            st.retract_reason = ""
            facts[fact_id] = st
            continue
        m = _RETRACT_RE.match(line)
        if m:
            fact_id = m.group(1)
            reason = m.group(2) or ""
            st = facts.get(fact_id) or FactState(fact_id=fact_id)
            st.retracted = True
            st.retract_reason = reason
            facts[fact_id] = st


# --------------------------------------------------------------------
# Verb-prefixed output parser: LLM emits one @VERB op per line covering
# the whole DSL surface. Python inflates each verb to the full DSL line.
# See src/supergraph/bonsai_dsl_prompt.txt for the contract.
#
# Line format: `@<VERB> <arg1> [arg2...]`. Lines without a leading `@`
# are silently dropped - English reasoning, fences, and <think> leaks are
# inert at parser level, so the model can drift safely without corrupting
# the emitted DSL.
#
# Verbs fall into three groups:
#   1. Fact-state (UPSERT / BELIEF / RETRACT): populate entities / beliefs /
#      retracts slots so _synthesize_dsl can auto-wire mention edges and
#      cross-message belief identity works.
#   2. Edge (EDGE): pre-renders a CREATE EDGE line.
#   3. Retrieval, walks, vault, sys ops: each pre-renders one full DSL
#      line directly.
#
# Groups 2 and 3 accumulate in turn.statements and get appended verbatim
# after the mention wiring and fact updates.
# --------------------------------------------------------------------


@dataclass
class ParsedTurn:
    """Parsed structured output of one @-verb LLM call.

    entities holds (slug, surface_name) tuples - the slug is the bare
    identifier the model emitted (without the legacy ``ent:`` prefix),
    surface_name is the human-readable form ("Alice", "OpenAI"). The
    synthesizer turns each into a mention node + a refers_to edge to a
    resolved entity (see ``supergraph.entity_resolver``).

    entity_edges holds (from_slug, to_slug, kind) tuples for @EDGE
    output between two entity slugs. Synthesizer maps each slug to the
    resolved entity_id and emits the edge between entity nodes.
    Edge handlers append here instead of `statements` so the slug ->
    entity_id rewrite happens after resolver runs.
    """

    entities: list[tuple[str, str]] = field(default_factory=list)
    entity_edges: list[tuple[str, str, str]] = field(default_factory=list)
    beliefs: list[tuple[str, str]] = field(default_factory=list)
    retracts: list[str] = field(default_factory=list)
    statements: list[str] = field(default_factory=list)


def _dsl_escape(s: str) -> str:
    """Escape a Python string for safe embedding inside a DSL "..." literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _h_upsert(turn: ParsedTurn, ln: str) -> None:
    """U <slug> <Name ...>

    Stores (slug, surface_name). The synthesizer turns this into a
    mention node + entity node (via the resolver) + refers_to edge.
    Slug is stored bare; legacy ``ent:`` prefix on input is stripped
    so the model can emit either form.
    """
    parts = ln.split(None, 2)
    if len(parts) < 3:
        return
    slug = parts[1].removeprefix("ent:").strip('"')
    name = parts[2].strip().strip('"')
    if slug and name:
        turn.entities.append((slug, name))


def _h_fact(turn: ParsedTurn, ln: str) -> None:
    """F <topic> <value ...>"""
    parts = ln.split(None, 2)
    if len(parts) < 3:
        return
    topic = parts[1].removeprefix("fact:").strip('"')
    value = parts[2].strip().strip('"')
    if topic and value:
        turn.beliefs.append((f"fact:{topic}", value))


def _h_drop(turn: ParsedTurn, ln: str) -> None:
    """D <topic>"""
    parts = ln.split()
    if len(parts) < 2:
        return
    topic = parts[1].removeprefix("fact:").strip('"')
    if topic:
        turn.retracts.append(f"fact:{topic}")


def _h_edge(turn: ParsedTurn, ln: str) -> None:
    """E <from_id> <to_id> <kind>  (ids may include legacy ent:/fact: prefix)

    Pushes (from_slug, to_slug, kind) to ``turn.entity_edges`` so the
    synthesizer can map each slug to the resolved entity_id before
    materializing the edge. The legacy ``ent:`` prefix is stripped so
    the slug can be looked up in the per-turn slug -> entity map. Bare
    fact:/msg: ids are passed through unchanged - they reference nodes
    that already exist by literal id.
    """
    parts = ln.split()
    if len(parts) < 4:
        return
    from_id = parts[1].strip('"')
    to_id = parts[2].strip('"')
    kind = parts[3].strip('"')
    if not (from_id and to_id and kind):
        return
    # Strip legacy "ent:" prefix; remap to entity ids during synthesis.
    from_slug = from_id.removeprefix("ent:")
    to_slug = to_id.removeprefix("ent:")
    turn.entity_edges.append((from_slug, to_slug, kind))


def _h_query(template: str):
    """Factory for verbs whose argument is free-text (wrapped in DSL quotes)."""
    def h(turn: ParsedTurn, ln: str) -> None:
        parts = ln.split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            return
        q = _dsl_escape(parts[1].strip())
        turn.statements.append(template.replace("{q}", q))
    return h


def _h_walk(template: str):
    """Factory for verbs whose argument is a single anchor id."""
    def h(turn: ParsedTurn, ln: str) -> None:
        parts = ln.split()
        if len(parts) < 2:
            return
        anchor = parts[1].strip('"')
        if anchor:
            turn.statements.append(template.replace("{id}", _dsl_escape(anchor)))
    return h


def _h_plain(template: str):
    """Factory for zero-arg verbs (SC/SH/ST/VS)."""
    def h(turn: ParsedTurn, _ln: str) -> None:
        turn.statements.append(template)
    return h


def _h_snapshot(turn: ParsedTurn, ln: str) -> None:
    """@SS [name]  ->  SYS SNAPSHOT "name".

    Grammar requires a name. If the model didn't supply one, auto-fill with
    a UTC timestamp so the emission is always parseable.
    """
    from datetime import datetime, timezone
    parts = ln.split(None, 1)
    if len(parts) >= 2 and parts[1].strip():
        name = parts[1].strip().strip('"')
    else:
        name = datetime.now(timezone.utc).strftime("snap-%Y%m%dT%H%M%SZ")
    turn.statements.append(f'SYS SNAPSHOT "{_dsl_escape(name)}"')


def _h_slug(template: str):
    """Factory for verbs taking a bare slug (auto-prefixed `ent:`).

    Used for node-level ops where the model emits `@DN my_node`; we quote +
    prefix to `ent:my_node` so the DSL parser accepts it. Also accepts a
    pre-prefixed id verbatim (`@DN ent:my_node`).
    """
    def h(turn: ParsedTurn, ln: str) -> None:
        parts = ln.split()
        if len(parts) < 2:
            return
        slug = parts[1].removeprefix("ent:").strip('"')
        if slug:
            turn.statements.append(template.replace("{slug}", _dsl_escape(slug)))
    return h


def _h_topic(template: str):
    """Factory for verbs taking a bare topic (auto-prefixed `fact:`)."""
    def h(turn: ParsedTurn, ln: str) -> None:
        parts = ln.split()
        if len(parts) < 2:
            return
        topic = parts[1].removeprefix("fact:").strip('"')
        if topic:
            turn.statements.append(template.replace("{topic}", _dsl_escape(topic)))
    return h


def _h_pair(template: str):
    """Factory for 2-anchor verbs (@PA, @SP, @CO). Template uses `{a}` / `{b}`.

    Ids are taken verbatim (caller supplies full `ent:`/`fact:` prefix).
    """
    def h(turn: ParsedTurn, ln: str) -> None:
        parts = ln.split()
        if len(parts) < 3:
            return
        a = parts[1].strip('"')
        b = parts[2].strip('"')
        if a and b:
            turn.statements.append(
                template.replace("{a}", _dsl_escape(a)).replace("{b}", _dsl_escape(b))
            )
    return h


def _h_update_node(turn: ParsedTurn, ln: str) -> None:
    """@UN slug field value ...  ->  UPDATE NODE "ent:slug" SET field = "value ..."

    Value is the rest-of-line (multi-word OK). Slug auto-prefixed `ent:`.
    """
    parts = ln.split(None, 3)
    if len(parts) < 4:
        return
    slug = parts[1].removeprefix("ent:").strip('"')
    field = parts[2].strip()
    value = parts[3].strip().strip('"')
    if not (slug and field and value):
        return
    turn.statements.append(
        f'UPDATE NODE "ent:{_dsl_escape(slug)}" SET {field} = "{_dsl_escape(value)}"'
    )


def _h_merge(turn: ParsedTurn, ln: str) -> None:
    """@M src dst  ->  MERGE NODE "ent:src" INTO "ent:dst"  (auto-prefix)"""
    parts = ln.split()
    if len(parts) < 3:
        return
    src = parts[1].removeprefix("ent:").strip('"')
    dst = parts[2].removeprefix("ent:").strip('"')
    if src and dst:
        turn.statements.append(
            f'MERGE NODE "ent:{_dsl_escape(src)}" INTO "ent:{_dsl_escape(dst)}"'
        )


def _h_raw(template: str):
    """Factory for verbs whose rest-of-line is a raw DSL body (passthrough).

    Used for verbs whose full grammar is too complex for positional encoding
    (MATCH patterns, AGGREGATE clauses, EVOLVE RULE conditions, WHERE-filtered
    bulk ops). The model emits the full DSL tail and Python just prefixes
    the leading keyword(s). No escaping applied.
    """
    def h(turn: ParsedTurn, ln: str) -> None:
        parts = ln.split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            return
        turn.statements.append(template.replace("{body}", parts[1].strip()))
    return h


def _h_update_edge(turn: ParsedTurn, ln: str) -> None:
    """@UE from to field value...  ->  UPDATE EDGE "from" -> "to" SET field = "value..."

    First two args are anchor ids (verbatim). Third is identifier. Rest is value.
    """
    parts = ln.split(None, 4)
    if len(parts) < 5:
        return
    a = parts[1].strip('"')
    b = parts[2].strip('"')
    field = parts[3].strip()
    value = parts[4].strip().strip('"')
    if a and b and field and value:
        turn.statements.append(
            f'UPDATE EDGE "{_dsl_escape(a)}" -> "{_dsl_escape(b)}" SET {field} = "{_dsl_escape(value)}"'
        )


def _h_increment(turn: ParsedTurn, ln: str) -> None:
    """@IC slug field num  ->  INCREMENT NODE "ent:slug" field BY num"""
    parts = ln.split()
    if len(parts) < 4:
        return
    slug = parts[1].removeprefix("ent:").strip('"')
    field = parts[2].strip()
    try:
        num = float(parts[3])
    except ValueError:
        return
    num_str = str(int(num)) if num == int(num) else str(num)
    if slug and field:
        turn.statements.append(
            f'INCREMENT NODE "ent:{_dsl_escape(slug)}" {field} BY {num_str}'
        )


def _h_propagate(turn: ParsedTurn, ln: str) -> None:
    """@PG anchor field depth  ->  PROPAGATE "anchor" FIELD field DEPTH n"""
    parts = ln.split()
    if len(parts) < 4:
        return
    anchor = parts[1].strip('"')
    field = parts[2].strip()
    try:
        depth = int(parts[3])
    except ValueError:
        return
    if anchor and field:
        turn.statements.append(
            f'PROPAGATE "{_dsl_escape(anchor)}" FIELD {field} DEPTH {depth}'
        )


def _h_describe(turn: ParsedTurn, ln: str) -> None:
    """@SD type name  ->  SYS DESCRIBE NODE|EDGE "name" """
    parts = ln.split()
    if len(parts) < 3:
        return
    t = parts[1].upper()
    if t not in ("NODE", "EDGE"):
        return
    name = parts[2].strip('"')
    if name:
        turn.statements.append(f'SYS DESCRIBE {t} "{_dsl_escape(name)}"')


def _h_unregister(turn: ParsedTurn, ln: str) -> None:
    """@SUR type name  ->  SYS UNREGISTER NODE|EDGE KIND "name" """
    parts = ln.split()
    if len(parts) < 3:
        return
    t = parts[1].upper()
    if t not in ("NODE", "EDGE"):
        return
    name = parts[2].strip('"')
    if name:
        turn.statements.append(f'SYS UNREGISTER {t} KIND "{_dsl_escape(name)}"')


def _h_contradictions(turn: ParsedTurn, ln: str) -> None:
    """@SCT field group  ->  SYS CONTRADICTIONS FIELD field GROUP BY group"""
    parts = ln.split()
    if len(parts) < 3:
        return
    field = parts[1].strip()
    group = parts[2].strip()
    if field and group:
        turn.statements.append(
            f'SYS CONTRADICTIONS FIELD {field} GROUP BY {group}'
        )


def _h_cron_add(turn: ParsedTurn, ln: str) -> None:
    """@CRA name schedule query...  ->  SYS CRON ADD "name" SCHEDULE "sched" QUERY "..."

    Uses shell-style quoting so cron expressions with spaces can be wrapped in
    quotes (`@CRA nightly "0 0 * * *" SYS STATS`). Everything after the
    schedule token becomes the query body.
    """
    import shlex
    try:
        tokens = shlex.split(ln)
    except ValueError:
        return
    if len(tokens) < 4:
        return
    name, schedule = tokens[1], tokens[2]
    query = " ".join(tokens[3:])
    if name and schedule and query:
        turn.statements.append(
            f'SYS CRON ADD "{_dsl_escape(name)}" '
            f'SCHEDULE "{_dsl_escape(schedule)}" '
            f'QUERY "{_dsl_escape(query)}"'
        )


def _h_optimize(turn: ParsedTurn, ln: str) -> None:
    """@SO [target]  ->  SYS OPTIMIZE [target]. target in {COMPACT,STRINGS,EDGES,VECTORS,BLOBS,CACHE}."""
    parts = ln.split()
    valid = {"COMPACT", "STRINGS", "EDGES", "VECTORS", "BLOBS", "CACHE"}
    if len(parts) >= 2:
        t = parts[1].strip().upper()
        if t in valid:
            turn.statements.append(f'SYS OPTIMIZE {t}')
    else:
        turn.statements.append('SYS OPTIMIZE')


def _h_clear(turn: ParsedTurn, ln: str) -> None:
    """@SCL target  ->  SYS CLEAR LOG|CACHE"""
    parts = ln.split()
    if len(parts) < 2:
        return
    t = parts[1].strip().upper()
    if t in ("LOG", "CACHE"):
        turn.statements.append(f'SYS CLEAR {t}')


def _h_wal(turn: ParsedTurn, ln: str) -> None:
    """@SWA action  ->  SYS WAL STATUS|REPLAY"""
    parts = ln.split()
    if len(parts) < 2:
        return
    a = parts[1].strip().upper()
    if a in ("STATUS", "REPLAY"):
        turn.statements.append(f'SYS WAL {a}')


def _h_vault_triplet(template: str):
    """Factory for @VW / @VAP: path + section + (multi-word) content."""
    def h(turn: ParsedTurn, ln: str) -> None:
        parts = ln.split(None, 3)
        if len(parts) < 4:
            return
        path = parts[1].strip('"')
        section = parts[2].strip('"')
        content = parts[3].strip().strip('"')
        if path and section and content:
            turn.statements.append(
                template.replace("{p}", _dsl_escape(path))
                        .replace("{s}", _dsl_escape(section))
                        .replace("{c}", _dsl_escape(content))
            )
    return h


def _h_nodes(turn: ParsedTurn, ln: str) -> None:
    """@NS [where-body]  ->  NODES [WHERE body] LIMIT 20"""
    parts = ln.split(None, 1)
    if len(parts) >= 2 and parts[1].strip():
        turn.statements.append(f'NODES WHERE {parts[1].strip()} LIMIT 20')
    else:
        turn.statements.append('NODES LIMIT 20')


# Handler instances reused across the short-code + English-keyword aliases.
# Building once and aliasing keeps the dispatch table lean and makes it
# obvious that `@RM` and `@REMEMBER` are the same handler, not two copies.

_H_RM = _h_query('REMEMBER "{q}" LIMIT 10')
_H_SM = _h_query('SIMILAR TO "{q}" LIMIT 10')
_H_LX = _h_query('LEXICAL SEARCH "{q}" LIMIT 10')
_H_AQ = _h_query('ANSWER "{q}"')

_H_RL = _h_walk('RECALL FROM "{id}" DEPTH 2')
_H_TR = _h_walk('TRAVERSE FROM "{id}" DEPTH 2')
_H_AN = _h_walk('ANCESTORS OF "{id}" DEPTH 3')
_H_DE = _h_walk('DESCENDANTS OF "{id}" DEPTH 3')
_H_SG = _h_walk('SUBGRAPH FROM "{id}" DEPTH 2')
_H_NO = _h_walk('NODE "{id}"')

_H_PA = _h_pair('PATH FROM "{a}" TO "{b}" MAX_DEPTH 3')
_H_PAS = _h_pair('PATHS FROM "{a}" TO "{b}" MAX_DEPTH 3')
_H_SP = _h_pair('SHORTEST PATH FROM "{a}" TO "{b}"')
_H_DI = _h_pair('DISTANCE FROM "{a}" TO "{b}" MAX_DEPTH 5')
_H_WSP = _h_pair('WEIGHTED SHORTEST PATH FROM "{a}" TO "{b}"')
_H_WDI = _h_pair('WEIGHTED DISTANCE FROM "{a}" TO "{b}"')
_H_CO = _h_pair('COMMON NEIGHBORS OF "{a}" AND "{b}"')
_H_EX = _h_pair('DELETE EDGE "{a}" -> "{b}"')

_H_DN = _h_slug('DELETE NODE "ent:{slug}"')
_H_FG = _h_slug('FORGET NODE "ent:{slug}"')
_H_CND = _h_slug('CONNECT NODE "ent:{slug}"')
_H_DEF = _h_slug('DELETE EDGES FROM "ent:{slug}"')
_H_DET = _h_slug('DELETE EDGES TO "ent:{slug}"')
_H_EF = _h_slug('EDGES FROM "ent:{slug}" LIMIT 20')
_H_ET = _h_slug('EDGES TO "ent:{slug}" LIMIT 20')

_H_CF = _h_topic('WHAT IF RETRACT "fact:{topic}"')

_H_MA = _h_raw('MATCH {body}')
_H_AG = _h_raw('AGGREGATE NODES {body}')
_H_UNS = _h_raw('UPDATE NODES WHERE {body}')
_H_DNS = _h_raw('DELETE NODES WHERE {body}')
_H_SRN = _h_raw('SYS REGISTER NODE KIND {body}')
_H_SRE = _h_raw('SYS REGISTER EDGE KIND {body}')
_H_EVR = _h_raw('SYS EVOLVE RULE {body}')

_H_VN = _h_query('VAULT NEW "{q}"')
_H_VR_READ = _h_query('VAULT READ "{q}"')
_H_VB = _h_query('VAULT BACKLINKS "{q}"')
_H_VQ = _h_query('VAULT SEARCH "{q}" LIMIT 10')
_H_VH = _h_query('VAULT ARCHIVE "{q}"')
_H_VW = _h_vault_triplet('VAULT WRITE "{p}" SECTION "{s}" CONTENT "{c}"')
_H_VAP = _h_vault_triplet('VAULT APPEND "{p}" SECTION "{s}" CONTENT "{c}"')

_H_BC = _h_query('BIND CONTEXT "{q}"')
_H_XC = _h_query('DISCARD CONTEXT "{q}"')
_H_IG = _h_query('INGEST "{q}"')
_H_SR = _h_query('SYS ROLLBACK TO "{q}"')
_H_SX = _h_query('SYS EXPLAIN REMEMBER "{q}"')

_H_PLAIN_COUNT_NODES = _h_plain('COUNT NODES')
_H_PLAIN_COUNT_EDGES = _h_plain('COUNT EDGES')

_H_PLAIN_COMPACT = _h_plain('SYS OPTIMIZE COMPACT')
_H_PLAIN_HEALTH = _h_plain('SYS HEALTH')
_H_PLAIN_STATS = _h_plain('SYS STATS')
_H_PLAIN_KINDS = _h_plain('SYS KINDS')
_H_PLAIN_EDGE_KINDS = _h_plain('SYS EDGE KINDS')
_H_PLAIN_EMBEDDERS = _h_plain('SYS EMBEDDERS')
_H_PLAIN_STATUS = _h_plain('SYS STATUS')
_H_PLAIN_SLOW = _h_plain('SYS SLOW QUERIES LIMIT 20')
_H_PLAIN_FREQUENT = _h_plain('SYS FREQUENT QUERIES LIMIT 20')
_H_PLAIN_FAILED = _h_plain('SYS FAILED QUERIES LIMIT 20')
_H_PLAIN_LOG = _h_plain('SYS LOG LIMIT 50')
_H_PLAIN_SNAPSHOTS = _h_plain('SYS SNAPSHOTS')
_H_PLAIN_VAULT_LIST = _h_plain('VAULT LIST')
_H_PLAIN_VAULT_DAILY = _h_plain('VAULT DAILY')
_H_PLAIN_VAULT_SYNC = _h_plain('VAULT SYNC')
_H_PLAIN_CHECKPOINT = _h_plain('SYS CHECKPOINT')
_H_PLAIN_REBUILD = _h_plain('SYS REBUILD INDICES')
_H_PLAIN_EXPIRE = _h_plain('SYS EXPIRE')
_H_PLAIN_DUPLICATES = _h_plain('SYS DUPLICATES')
_H_PLAIN_SYS_CONNECT = _h_plain('SYS CONNECT')
_H_PLAIN_CONSOLIDATE = _h_plain('SYS CONSOLIDATE')
_H_PLAIN_REEMBED = _h_plain('SYS REEMBED')
_H_PLAIN_RETAIN = _h_plain('SYS RETAIN')
_H_PLAIN_EVICT = _h_plain('SYS EVICT')
_H_PLAIN_CRON_LIST = _h_plain('SYS CRON LIST')
_H_PLAIN_EVOLVE_LIST = _h_plain('SYS EVOLVE LIST')
_H_PLAIN_EVOLVE_HISTORY = _h_plain('SYS EVOLVE HISTORY LIMIT 50')
_H_PLAIN_EVOLVE_RESET = _h_plain('SYS EVOLVE RESET')

_H_Q_CRON_DELETE = _h_query('SYS CRON DELETE "{q}"')
_H_Q_CRON_ENABLE = _h_query('SYS CRON ENABLE "{q}"')
_H_Q_CRON_DISABLE = _h_query('SYS CRON DISABLE "{q}"')
_H_Q_CRON_RUN = _h_query('SYS CRON RUN "{q}"')
_H_Q_EVOLVE_SHOW = _h_query('SYS EVOLVE SHOW "{q}"')
_H_Q_EVOLVE_ENABLE = _h_query('SYS EVOLVE ENABLE "{q}"')
_H_Q_EVOLVE_DISABLE = _h_query('SYS EVOLVE DISABLE "{q}"')
_H_Q_EVOLVE_DELETE = _h_query('SYS EVOLVE DELETE "{q}"')


_VERB_HANDLERS: dict[str, Any] = {
    # Ingest: entities, beliefs, retract. BELIEF/ASSERT/BELIEVE all map to
    # the same handler so the model can use whichever phrasing feels natural.
    "UPSERT": _h_upsert,
    "BELIEF": _h_fact, "BELIEVE": _h_fact, "ASSERT": _h_fact, "FACT": _h_fact,
    "RETRACT": _h_drop, "DROP": _h_drop,

    # Edges
    "EDGE": _h_edge, "CREATE_EDGE": _h_edge,
    "UPDATE_EDGE": _h_update_edge,
    "DELETE_EDGE": _H_EX,
    "DELETE_EDGES_FROM": _H_DEF,
    "DELETE_EDGES_TO": _H_DET,
    "EDGES_FROM": _H_EF,
    "EDGES_TO": _H_ET,

    # Node lifecycle
    "UPDATE_NODE": _h_update_node,
    "DELETE_NODE": _H_DN,
    "FORGET": _H_FG, "FORGET_NODE": _H_FG,
    "CONNECT_NODE": _H_CND,
    "MERGE": _h_merge, "MERGE_NODE": _h_merge,
    "INCREMENT": _h_increment,
    "PROPAGATE": _h_propagate,
    "COUNTERFACTUAL": _H_CF, "WHAT_IF": _H_CF,
    "NODE": _H_NO,
    "NODES": _h_nodes,

    # Bulk WHERE-filtered (raw passthrough)
    "UPDATE_NODES": _H_UNS,
    "DELETE_NODES": _H_DNS,

    # Retrieval (user asked a question)
    "REMEMBER": _H_RM,
    "SIMILAR": _H_SM, "SIMILAR_TO": _H_SM,
    "LEXICAL": _H_LX, "LEXICAL_SEARCH": _H_LX, "SEARCH": _H_LX,
    "ANSWER": _H_AQ,

    # Counts
    "COUNT_NODES": _H_PLAIN_COUNT_NODES,
    "COUNT_EDGES": _H_PLAIN_COUNT_EDGES,

    # Walks
    "RECALL": _H_RL,
    "TRAVERSE": _H_TR,
    "ANCESTORS": _H_AN,
    "DESCENDANTS": _H_DE,
    "SUBGRAPH": _H_SG,

    # Paths / distance
    "PATH": _H_PA,
    "PATHS": _H_PAS,
    "SHORTEST": _H_SP, "SHORTEST_PATH": _H_SP,
    "DISTANCE": _H_DI,
    "WEIGHTED_SHORTEST": _H_WSP, "WEIGHTED_SHORTEST_PATH": _H_WSP,
    "WEIGHTED_DISTANCE": _H_WDI,
    "COMMON": _H_CO, "COMMON_NEIGHBORS": _H_CO,

    # Pattern / aggregate (raw passthrough)
    "MATCH": _H_MA,
    "AGGREGATE": _H_AG,

    # Vault
    "VAULT_NEW": _H_VN,
    "VAULT_READ": _H_VR_READ,
    "VAULT_WRITE": _H_VW,
    "VAULT_APPEND": _H_VAP,
    "VAULT_LIST": _H_PLAIN_VAULT_LIST,
    "VAULT_BACKLINKS": _H_VB,
    "VAULT_SEARCH": _H_VQ,
    "VAULT_DAILY": _H_PLAIN_VAULT_DAILY,
    "VAULT_ARCHIVE": _H_VH,
    "VAULT_SYNC": _H_PLAIN_VAULT_SYNC,

    # Context + doc ingest
    "BIND_CONTEXT": _H_BC,
    "DISCARD_CONTEXT": _H_XC,
    "INGEST": _H_IG,

    # Snapshots / rollback / optimize
    "SNAPSHOT": _h_snapshot,
    "ROLLBACK": _H_SR,
    "SNAPSHOTS": _H_PLAIN_SNAPSHOTS,
    "COMPACT": _H_PLAIN_COMPACT,
    "OPTIMIZE": _h_optimize,

    # SYS introspection
    "HEALTH": _H_PLAIN_HEALTH,
    "STATS": _H_PLAIN_STATS,
    "KINDS": _H_PLAIN_KINDS,
    "EDGE_KINDS": _H_PLAIN_EDGE_KINDS,
    "EMBEDDERS": _H_PLAIN_EMBEDDERS,
    "STATUS": _H_PLAIN_STATUS,
    "SLOW_QUERIES": _H_PLAIN_SLOW,
    "FREQUENT_QUERIES": _H_PLAIN_FREQUENT,
    "FAILED_QUERIES": _H_PLAIN_FAILED,
    "LOG": _H_PLAIN_LOG,
    "EXPLAIN": _H_SX,
    "DESCRIBE": _h_describe,

    # SYS schema registration
    "REGISTER_NODE": _H_SRN, "REGISTER_NODE_KIND": _H_SRN,
    "REGISTER_EDGE": _H_SRE, "REGISTER_EDGE_KIND": _H_SRE,
    "UNREGISTER": _h_unregister,

    # SYS maintenance (admin)
    "CHECKPOINT": _H_PLAIN_CHECKPOINT,
    "REBUILD": _H_PLAIN_REBUILD,
    "CLEAR": _h_clear,
    "WAL": _h_wal,
    "EXPIRE": _H_PLAIN_EXPIRE,
    "CONTRADICTIONS": _h_contradictions,
    "DUPLICATES": _H_PLAIN_DUPLICATES,
    "CONNECT_ALL": _H_PLAIN_SYS_CONNECT,
    "CONSOLIDATE": _H_PLAIN_CONSOLIDATE,
    "REEMBED": _H_PLAIN_REEMBED,
    "RETAIN": _H_PLAIN_RETAIN,
    "EVICT": _H_PLAIN_EVICT,

    # Cron
    "CRON_ADD": _h_cron_add,
    "CRON_DELETE": _H_Q_CRON_DELETE,
    "CRON_ENABLE": _H_Q_CRON_ENABLE,
    "CRON_DISABLE": _H_Q_CRON_DISABLE,
    "CRON_LIST": _H_PLAIN_CRON_LIST,
    "CRON_RUN": _H_Q_CRON_RUN,

    # Evolve (metacognitive rules)
    "EVOLVE_LIST": _H_PLAIN_EVOLVE_LIST,
    "EVOLVE_SHOW": _H_Q_EVOLVE_SHOW,
    "EVOLVE_ENABLE": _H_Q_EVOLVE_ENABLE,
    "EVOLVE_DISABLE": _H_Q_EVOLVE_DISABLE,
    "EVOLVE_DELETE": _H_Q_EVOLVE_DELETE,
    "EVOLVE_HISTORY": _H_PLAIN_EVOLVE_HISTORY,
    "EVOLVE_RESET": _H_PLAIN_EVOLVE_RESET,
    "EVOLVE_RULE": _H_EVR,
}


def _parse_verb_output(cleaned: str) -> ParsedTurn:
    """Parse v6 @-prefixed verb-positional output.

    Every op line starts with `@`. Python drops any line that doesn't,
    making English drift / reasoning leaks / markdown fences inert at the
    parser level. For valid lines, strip the `@`, dispatch the verb to its
    handler. @U/@F/@D populate the entities/beliefs/retracts slots for
    fact-state wiring; all other verbs render directly to pre-built DSL
    lines in turn.statements.

    Tolerant: accepts lowercase verbs, extra whitespace after `@`, trailing
    colons on the verb token (seen in reasoning models that mimic chat
    prefixes). Unknown verbs drop silently.
    """
    turn = ParsedTurn()
    for raw_ln in cleaned.splitlines():
        ln = raw_ln.strip()
        if not ln or _FENCE_RE.match(ln):
            continue
        if not ln.startswith("@"):
            continue
        payload = ln[1:].lstrip()
        if not payload:
            continue
        head = payload.split(maxsplit=1)[0]
        verb = head.upper().rstrip(":")
        handler = _VERB_HANDLERS.get(verb)
        if handler is None:
            continue
        handler(turn, payload)
    return turn


def _synthesize_dsl(
    turn: ParsedTurn,
    *,
    msg_id: str,
    session_id: str,
    role: str,
    text: str,
    gs: Any | None = None,
) -> list[str]:
    """Render parsed @-verbs to DSL. ``gs`` enables resolver lookup;
    when None (dry-run, tests) every mention mints a fresh entity."""
    from supergraph.entity_resolver import (
        EDGE_REFERS_TO, KIND_ENTITY, KIND_MENTION,
        make_entity_id, make_mention_id,
        resolve_and_create_entity, resolve_mention,
    )

    out: list[str] = []
    text_esc = _dsl_escape(text)
    session_esc = _dsl_escape(session_id)
    role_esc = _dsl_escape(role)
    msg_esc = _dsl_escape(msg_id)

    # `content` mirrors `DOCUMENT` so adapters that scan typed columns
    # (deterministic NER baseline) compare apples-to-apples with the
    # vector + BM25 pipeline that reads `DOCUMENT`.
    out.append(
        f'CREATE NODE "{msg_esc}" kind = "message" '
        f'session = "{session_esc}" role = "{role_esc}" '
        f'content = "{text_esc}" '
        f'DOCUMENT "{text_esc}"'
    )

    slug_to_entity: dict[str, str] = {}

    seen_slugs: set[str] = set()
    for occurrence, (slug, name) in enumerate(turn.entities):
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        mention_id = make_mention_id(msg_id, slug, occurrence)
        mention_esc = _dsl_escape(mention_id)
        name_esc = _dsl_escape(name)

        if gs is not None:
            # Atomic resolve+create under process lock - protects
            # against the race where two concurrent ingests of the
            # same name both observe zero candidates and both mint
            # fresh entities. With the wrapper, by the time the
            # second caller's resolve() runs, the first caller's
            # entity is already in the store.
            try:
                entity_id, confidence, is_new = resolve_and_create_entity(
                    gs, surface_name=name, context=text,
                )
            except Exception as e:  # pragma: no cover
                _log.warning("entity_resolver failed for %r: %s", name, e)
                entity_id = make_entity_id()
                is_new = True
                confidence = 1.0
        else:
            entity_id = make_entity_id()
            is_new = True
            confidence = 1.0

        slug_to_entity[slug] = entity_id
        entity_esc = _dsl_escape(entity_id)

        out.append(
            f'CREATE NODE "{mention_esc}" kind = "{KIND_MENTION}" '
            f'surface_name = "{name_esc}" source_msg = "{msg_esc}" '
            f'session = "{session_esc}" '
            f'DOCUMENT "{name_esc} | {text_esc}"'
        )

        # When gs is None (dry-run / tests) the entity hasn't been
        # written yet and we still need to emit a CREATE NODE for it.
        # When gs is real, resolve_and_create_entity already wrote it
        # if needed; we just bump mention_count.
        if gs is None and is_new:
            out.append(
                f'CREATE NODE "{entity_esc}" kind = "{KIND_ENTITY}" '
                f'canonical_name = "{name_esc}" mention_count = 1 '
                f'context = "{text_esc}" '
                f'DOCUMENT "{name_esc} | {text_esc}"'
            )
        else:
            out.append(
                f'INCREMENT NODE "{entity_esc}" mention_count BY 1'
            )

        out.append(
            f'CREATE EDGE "{mention_esc}" -> "{entity_esc}" '
            f'kind = "{EDGE_REFERS_TO}" confidence = {confidence:.3f}'
        )
        out.append(
            f'CREATE EDGE "{msg_esc}" -> "{mention_esc}" kind = "mentions"'
        )

    for fact_id in turn.retracts:
        out.append(
            f'RETRACT "{_dsl_escape(fact_id)}" REASON "superseded by {msg_esc}"'
        )

    for fact_id, value in turn.beliefs:
        out.append(
            f'ASSERT "{_dsl_escape(fact_id)}" kind = "belief" '
            f'value = "{_dsl_escape(value)}" CONFIDENCE 0.9 SOURCE "{msg_esc}"'
        )

    for from_slug, to_slug, edge_kind in turn.entity_edges:
        from_eid = slug_to_entity.get(from_slug)
        to_eid = slug_to_entity.get(to_slug)
        if not (from_eid and to_eid):
            _log.warning(
                "bonsai: dropping @EDGE %r -> %r (slug not in turn map)",
                from_slug, to_slug,
            )
            continue
        out.append(
            f'CREATE EDGE "{_dsl_escape(from_eid)}" -> "{_dsl_escape(to_eid)}" '
            f'kind = "{_dsl_escape(edge_kind)}"'
        )

    # Retrieval verbs (@RECALL/@PATH/@ANCESTORS/...) emit literal
    # `ent:slug` ids. After the refactor those nodes do not exist;
    # rewrite each reference to the matching entity_id from the turn
    # map, falling back to the literal slug when unknown so debug
    # output remains traceable.
    # Build a cross-turn slug -> entity_id map so walk/path verbs can resolve
    # entities created in earlier turns (graph ids are content-hashed, not
    # slugs, so the model's `ent:marie_curie` only matches via slugified
    # canonical_name). Only paid for when statements actually carry ent: refs.
    graph_slugs: dict[str, str] | None = None
    if gs is not None and any("ent:" in s for s in turn.statements):
        from supergraph.ingest.entity_extract import slug as _ent_slug
        graph_slugs = {}
        try:
            for n in gs.execute('NODES WHERE kind = "entity"').data or []:
                cn = n.get("canonical_name")
                if cn:
                    graph_slugs.setdefault(_ent_slug(cn), n["id"])
        except Exception:
            graph_slugs = None

    for stmt in turn.statements:
        out.append(_rewrite_ent_refs(stmt, slug_to_entity, graph_slugs))

    return out


def _rewrite_ent_refs(
    stmt: str,
    slug_to_entity: dict[str, str],
    graph_slugs: dict[str, str] | None = None,
) -> str:
    """Rewrite ``ent:slug`` references to real entity ids.

    ``_ENT_FROM_ID_RE`` captures the WITH-prefix form (``ent:marie_curie``),
    so we strip the prefix before looking the bare slug up. Resolution order:
    this turn's freshly-minted entities (``slug_to_entity``) -> entities
    already in the graph (``graph_slugs``, keyed by slugified canonical_name)
    -> unresolved. Unresolved refs fall back to a SINGLE ``ent:`` prefix; the
    pre-fix code emitted ``f'"ent:{slug}"'`` with ``slug`` still carrying its
    own ``ent:``, producing the ``ent:ent:`` double-prefix that made every
    cross-turn walk/path query miss (audit 2026-06-13).
    """
    def _sub(match: re.Match) -> str:
        ref = match.group(1)
        bare = ref[4:] if ref.startswith("ent:") else ref
        eid = slug_to_entity.get(bare)
        if not eid and graph_slugs:
            eid = graph_slugs.get(bare)
        return f'"{eid}"' if eid else f'"ent:{bare}"'
    return _ENT_FROM_ID_RE.sub(_sub, stmt)


def _render_known_facts_block(facts: dict[str, FactState], max_facts: int = 40) -> str:
    """Format non-retracted facts into a block the LLM reads before the input.

    Placed in the USER message (not the system prompt) to keep the system
    prompt byte-identical across calls for llama.cpp KV-cache prefix reuse.

    Retracted facts are hidden so the model cannot re-assert a superseded
    belief. If the dict grows beyond `max_facts`, the most-recently-touched
    entries win (dict preserves insertion order since Python 3.7).
    """
    alive = [f for f in facts.values() if not f.retracted]
    if not alive:
        return ""
    alive = alive[-max_facts:]
    lines = [
        "### KNOWN FACTS (reuse these fact_ids; emit RETRACT + ASSERT to update)",
    ]
    for f in alive:
        bits = [f'[{f.fact_id}]']
        if f.kind:
            bits.append(f'kind="{f.kind}"')
        if f.value:
            bits.append(f'value="{f.value}"')
        bits.append(f'confidence={f.confidence:.2f}')
        if f.source:
            bits.append(f'source="{f.source}"')
        lines.append(" ".join(bits))
    return "\n".join(lines)


# --------------------------------------------------------------------
# Ingestor
# --------------------------------------------------------------------

# The verb-prefixed prompt lives inside the package so it ships with the
# wheel and does not depend on the tools/skills/ tree being present at runtime.
# - full:  covers the complete grammar (ingest + edges + node ops + queries +
#          walks + paths + vault + snapshots + sys admin + cron + evolve).
# - lite:  only ingest (UPSERT/BELIEF/RETRACT/EDGE) + retrieval (REMEMBER /
#          SIMILAR / LEXICAL / ANSWER + walks + paths). Smaller prompt = less
#          model confusion when the caller never uses admin ops.
_DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "bonsai_dsl_prompt.txt"
_DEFAULT_LITE_PROMPT_PATH = Path(__file__).resolve().parent / "bonsai_dsl_prompt_lite.txt"


class BonsaiIngestor:
    """NL -> DSL via a local llama.cpp GGUF, with correctness guards.

    Parameters
    ----------
    model_path : str | Path | None
        Path to a .gguf file. When omitted, the GGUF is auto-resolved via
        ``supergraph._models.resolve_bonsai_gguf`` - it is read from the
        HuggingFace cache or, on first use, downloaded into it. The matching
        manifest under the same directory is not required; this class talks
        to llama.cpp directly.
    quant : str | None
        Quantization to resolve when ``model_path`` is not provided. Defaults
        to ``$SUPERGRAPH_BONSAI_QUANT`` or ``"TQ1_0"``. Ignored when
        ``model_path`` is given.
    gs : SuperGraph | None
        Target store. Required for non-dry-run ingests. Dry-runs don't need one.
    skill_path : str | Path | None
        Prompt file. Defaults to tools/skills/supergraph-bonsai-dsl/SKILL.md.
    n_ctx : int
        Context window. 2048 is enough for ~500-token skill + 200-token user
        message + 400-token output + headroom.
    n_threads : int | None
        Physical core count works best on memory-bandwidth-bound CPU inference.
        Defaults to os.cpu_count() // 2 via llama.cpp.
    chat_format : str
        Matches the GGUF. Qwen3-based Bonsai works with 'qwen'.
    max_output_tokens : int
        Hard cap per call. Generation stops either here or at the model's
        natural stop.
    temperature : float
        Default 0.0 for reproducible DSL.
    """

    # Headroom we leave between (prompt + output) and n_ctx. Below this we
    # force a reset so llama.cpp never auto-evicts the skill prefix.
    _CTX_HEADROOM = 128

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        quant: str | None = None,
        gs: Any | None = None,
        skill_path: str | Path | None = None,
        n_ctx: int | None = None,
        n_batch: int = 512,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        chat_format: str = "qwen",
        max_output_tokens: int = 256,
        temperature: float = 0.0,
        kv_cache_path: str | Path | None = None,
        flash_attn: bool = False,
        ner_model_dir: str | Path | None = None,
        ner_score_threshold: float = 0.7,
        ner_max_hints: int = 6,
    ) -> None:
        if model_path is None:
            from supergraph._models import resolve_bonsai_gguf
            self._model_path = resolve_bonsai_gguf(quant)
        else:
            self._model_path = Path(model_path)
            if not self._model_path.exists():
                raise FileNotFoundError(f"bonsai model not found: {self._model_path}")
        self._gs = gs
        self._skill_path = Path(skill_path) if skill_path else _DEFAULT_PROMPT_PATH
        # Dense user turns can legitimately need 10-15 ops (30-100 tokens).
        # Cap high enough to cover that. Post-op English drift is inert
        # because the parser ignores non-@ lines, so over-provisioning only
        # costs wall time on bad turns, never correctness.
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._chat_format = chat_format
        self._n_threads = n_threads
        # n_batch defaults to llama.cpp's 512. On CPU, bigger batches saturate
        # memory bandwidth and actually slow things down (measured: n_batch=2048
        # was 18% slower overall than 512 on this hardware). Kept exposed as a
        # kwarg for GPU callers where bigger batches can help.
        self._n_batch = n_batch
        self._flash_attn = flash_attn
        # 0 = CPU only (default). -1 = offload all layers to GPU. Positive
        # int = offload that many layers. Requires a CUDA/Metal/Vulkan build
        # of llama-cpp-python; the CPU-only wheel silently ignores it.
        self._n_gpu_layers = n_gpu_layers

        self._skill_text = ""
        self._skill_fingerprint = ""
        self._system_prompt = ""
        self._reload_skill()

        # Auto-pick n_ctx based on actual prompt size unless caller pinned
        # one explicitly. Full prompt (~1700 tokens) needs 4096; lite prompt
        # (~600 tokens) fits 2048 comfortably and halves KV cache RAM + load
        # time. Typical user-message budget: 300 tokens (includes KNOWN FACTS
        # block when present).
        if n_ctx is None:
            _USER_MSG_BUDGET = 300
            needed = (
                self._estimate_tokens(self._system_prompt)
                + _USER_MSG_BUDGET
                + self._max_output_tokens
                + self._CTX_HEADROOM
            )
            for candidate in (2048, 4096, 8192, 16384, 32768):
                if needed <= candidate:
                    n_ctx = candidate
                    break
            else:
                n_ctx = 32768
            _log.info(
                "bonsai: auto-picked n_ctx=%d (prompt~%d + user~%d + output=%d + headroom=%d)",
                n_ctx,
                self._estimate_tokens(self._system_prompt),
                _USER_MSG_BUDGET,
                self._max_output_tokens,
                self._CTX_HEADROOM,
            )
        self._n_ctx = n_ctx

        self._llm: Any | None = None
        self._lock = threading.Lock()

        # Cross-message belief tracking. fact_id -> FactState. Fed into the
        # user message of the next ingest so the model reuses ids.
        self._facts: dict[str, FactState] = {}

        # Optional persistent KV cache. Eliminates the ~10s cold penalty on
        # process restarts. File holds a pickled (meta, LlamaState) tuple;
        # meta guards against loading stale state when the skill or config
        # changed since the cache was written.
        self._kv_cache_path = Path(kv_cache_path) if kv_cache_path else None

        # Optional NER hint feed. When set, every ingest runs the input
        # through the deterministic TinyBERT extractor and prepends a
        # `[ner:a,b,c]` line to the user message. The model treats it as a
        # noisy candidate list - it can keep, drop, or augment - so blind NER
        # misses do not silently propagate into the graph. None disables.
        self._ner_model_dir = Path(ner_model_dir) if ner_model_dir else None
        self._ner_score_threshold = ner_score_threshold
        self._ner_max_hints = max(0, int(ner_max_hints))
        self._ner_disabled = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _reload_skill(self) -> None:
        """Read skill from disk, compute fingerprint, pin into system prompt.

        Pinning the fingerprint into the prompt means if the file changes on
        disk the system-prompt prefix changes too, which naturally invalidates
        the llama.cpp prefix-match KV cache without us having to call reset.
        """
        if not self._skill_path.exists():
            raise FileNotFoundError(f"bonsai skill not found: {self._skill_path}")
        body = self._skill_path.read_text()
        if body.startswith("---"):
            _, _, body = body.partition("---")
            _, _, body = body.partition("---")
            body = body.strip()
        self._skill_text = body
        self._skill_fingerprint = hashlib.sha256(body.encode()).hexdigest()[:12]
        self._system_prompt = f"# skill-sha256={self._skill_fingerprint}\n\n{body}"

    def _kv_meta(self) -> dict[str, Any]:
        """What the current config looks like. Written alongside the KV cache
        so we can refuse to load state if any of these changed."""
        return {
            "model_path": str(self._model_path),
            "model_size_bytes": self._model_path.stat().st_size,
            "skill_fingerprint": self._skill_fingerprint,
            "n_ctx": self._n_ctx,
            "chat_format": self._chat_format,
        }

    def _try_load_kv_cache(self, llm: Any) -> bool:
        """Load a persisted KV cache into `llm` if one exists and is valid.

        Returns True on successful load, False otherwise. Invalid cache is
        silently ignored - the caller warms up normally.
        """
        if not self._kv_cache_path or not self._kv_cache_path.exists():
            return False
        import pickle

        try:
            with self._kv_cache_path.open("rb") as f:
                payload = pickle.load(f)
        except Exception as err:
            _log.warning("bonsai: KV cache unreadable (%s); skipping", err)
            return False

        meta = payload.get("meta") if isinstance(payload, dict) else None
        state = payload.get("state") if isinstance(payload, dict) else None
        if not meta or state is None:
            _log.warning("bonsai: KV cache shape invalid; skipping")
            return False

        cur = self._kv_meta()
        if meta != cur:
            diff = {k: (meta.get(k), cur.get(k)) for k in cur if meta.get(k) != cur.get(k)}
            _log.info(
                "bonsai: KV cache stale (diff=%s); warming fresh",
                diff,
            )
            return False

        try:
            llm.load_state(state)
        except Exception as err:
            _log.warning("bonsai: KV cache load_state failed (%s); warming fresh", err)
            return False

        _log.info("bonsai: KV cache loaded from %s (skipped warmup)", self._kv_cache_path)
        return True

    def save_kv_cache(self) -> None:
        """Persist the current Llama instance's KV state to `kv_cache_path`.

        Call after `warmup()` (or after one real ingest) so the skill-prefix
        tokens are in the cache. The file is (meta, LlamaState) pickled.

        No-op if kv_cache_path was not configured or the Llama hasn't been
        constructed yet.
        """
        if not self._kv_cache_path or self._llm is None:
            return
        import pickle

        state = self._llm.save_state()
        self._kv_cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._kv_cache_path.with_suffix(self._kv_cache_path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump({"meta": self._kv_meta(), "state": state}, f)
        tmp.replace(self._kv_cache_path)
        _log.info(
            "bonsai: KV cache saved to %s (%.1f MB)",
            self._kv_cache_path,
            self._kv_cache_path.stat().st_size / 1e6,
        )

    def _ensure_llm(self) -> Any:
        """Lazy-load the Llama instance on first use."""
        if self._llm is not None:
            return self._llm
        from llama_cpp import Llama
        kwargs: dict[str, Any] = {
            "model_path": str(self._model_path),
            "n_ctx": self._n_ctx,
            "n_batch": self._n_batch,
            "n_gpu_layers": self._n_gpu_layers,
            "chat_format": self._chat_format,
            "flash_attn": self._flash_attn,
            "verbose": False,
        }
        if self._n_threads is not None:
            kwargs["n_threads"] = self._n_threads
        _log.info(
            "bonsai: loading %s n_ctx=%d n_batch=%d gpu_layers=%d flash_attn=%s "
            "threads=%s chat_format=%s",
            self._model_path.name, self._n_ctx, self._n_batch,
            self._n_gpu_layers, self._flash_attn, self._n_threads,
            self._chat_format,
        )
        self._llm = Llama(**kwargs)
        self._try_load_kv_cache(self._llm)
        return self._llm

    def reset(self) -> None:
        """Drop the Llama instance so the next call reloads from scratch.

        Use when the skill file changed and you want to force a cold start,
        or when KV state is suspected corrupt (e.g. after a thread crash).
        Automatic: only called from internal guards.
        """
        with self._lock:
            self._llm = None

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def skill_fingerprint(self) -> str:
        """12-hex-char sha256 prefix of the loaded skill text. Stable across
        processes for the same skill bytes. Emitted in every ingest log line."""
        return self._skill_fingerprint

    @property
    def facts(self) -> dict[str, FactState]:
        """Live fact state accumulated from successful ingests. Read-only view.

        Use reset_facts() to clear. Dry-run ingests don't update this.
        """
        return dict(self._facts)

    def reset_facts(self) -> None:
        """Clear the running fact state so the next ingest starts fresh."""
        with self._lock:
            self._facts.clear()

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Process the system prompt once so the skill-prefix KV is warm.

        Optional - the first real ingest pays this cost anyway. Separate
        because long-running daemons want the cost up-front, not on the
        first user-facing request.
        """
        llm = self._ensure_llm()
        with self._lock:
            llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": "ready"},
                ],
                max_tokens=1,
                temperature=0.0,
            )

    def ingest(
        self,
        text: str,
        *,
        msg_id: str | None = None,
        session_id: str = "default",
        role: str = "user",
        dry_run: bool = False,
    ) -> IngestResult:
        """Convert `text` to DSL statements and (optionally) execute them.

        The LLM emits @-verb lines; Python synthesizes the full DSL. The
        caller must pass `msg_id` (and may override `session_id` / `role`);
        these become the identifiers in the synthesized CREATE NODE /
        CREATE EDGE statements for mentions-edge wiring.

        `dry_run=True` returns the DSL without touching the store.
        """
        if not text or not text.strip():
            raise IngestEmpty("input text is empty or whitespace-only")
        if not dry_run and self._gs is None:
            raise ValueError("ingest requires a SuperGraph (pass gs=...) or dry_run=True")
        if not msg_id:
            raise ValueError(
                "ingest requires an explicit msg_id "
                "(DSL synthesis needs the exact CREATE NODE id)"
            )

        self._reload_skill()
        with self._lock:
            return self._ingest_locked(
                text,
                msg_id=msg_id,
                session_id=session_id,
                role=role,
                dry_run=dry_run,
            )

    def _ner_hints(self, text: str) -> str:
        """Run TinyBERT NER on `text` and format a one-line hint.

        Returns ``"[ner:a,b,c]"`` or ``""``. Failure modes are silent: any
        extractor error disables NER for the rest of this ingestor's life
        rather than crashing the LLM call. Empty results return "" so we
        do not pay the prompt-token cost of a useless hint.
        """
        if not self._ner_model_dir or self._ner_disabled or self._ner_max_hints == 0:
            return ""
        try:
            from supergraph.ingest.entity_extract import extract_entities
            ents = extract_entities(
                text,
                model_dir=self._ner_model_dir,
                score_threshold=self._ner_score_threshold,
            )
        except Exception as e:
            _log.warning("bonsai: NER hint extraction failed (%s); disabling for this ingestor", e)
            self._ner_disabled = True
            return ""
        if not ents:
            return ""
        seen: set[str] = set()
        names: list[str] = []
        for e in ents:
            nm = (getattr(e, "text", "") or "").strip()
            if not nm:
                continue
            key = nm.lower()
            if key in seen:
                continue
            seen.add(key)
            names.append(nm)
            if len(names) >= self._ner_max_hints:
                break
        if not names:
            return ""
        return f"[ner:{','.join(names)}]"

    def _ingest_locked(
        self,
        text: str,
        *,
        msg_id: str | None,
        session_id: str,
        role: str,
        dry_run: bool,
    ) -> IngestResult:
        t0 = time.perf_counter()
        llm = self._ensure_llm()

        # Compose the user message: prior facts block + optional NER hint
        # line + user text. System prompt stays byte-identical so the skill
        # KV cache stays warm.
        facts_block = _render_known_facts_block(self._facts)
        ner_hints = self._ner_hints(text)
        parts = [p for p in (facts_block, ner_hints, text) if p]
        user_msg = "\n\n".join(parts) if parts else text

        est = self._estimate_tokens(self._system_prompt) + self._estimate_tokens(user_msg)
        budget = est + self._max_output_tokens
        if budget > self._n_ctx - self._CTX_HEADROOM:
            if est + self._max_output_tokens > self._n_ctx - self._CTX_HEADROOM:
                raise IngestOverflow(
                    f"prompt+output ({budget}) exceeds n_ctx-headroom "
                    f"({self._n_ctx - self._CTX_HEADROOM}); increase n_ctx or "
                    f"shorten input"
                )
            _log.warning("bonsai: KV would overflow, resetting before ingest")
            self._llm = None
            llm = self._ensure_llm()

        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=self._max_output_tokens,
            temperature=self._temperature,
        )
        raw = response["choices"][0]["message"]["content"] or ""

        cleaned = _strip_think(raw)
        if not cleaned:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            self._log_event(text, raw, [], 0, [], [], [], duration_ms, dry_run)
            raise IngestEmpty(
                "LLM returned empty or <think>-only output. "
                f"raw={raw!r}"
            )

        assert msg_id is not None  # guarded in ingest()
        turn = _parse_verb_output(cleaned)
        deduped = _synthesize_dsl(
            turn, msg_id=msg_id, session_id=session_id, role=role, text=text,
            gs=None if dry_run else self._gs,
        )
        dup_dropped: list[tuple[str, str]] = []

        from supergraph.dsl.parser import parse as _dsl_parse

        valid: list[str] = []
        rejected: list[tuple[str, str]] = list(dup_dropped)
        for ln in deduped:
            try:
                _dsl_parse(ln)
                valid.append(ln)
            except Exception as err:
                rejected.append((ln, f"parse error: {err}"))

        entities_new: list[str] = []
        beliefs_changed: list[tuple[str, str]] = []
        for ln in valid:
            if _UPSERT_RE.match(ln):
                m = _ENT_FROM_ID_RE.search(ln)
                if m:
                    entities_new.append(m.group(1))
            elif _ASSERT_RE.match(ln):
                m = _ASSERT_RE.match(ln)
                if m:
                    beliefs_changed.append((m.group(1), "assert"))
            elif _RETRACT_RE.match(ln):
                m = _RETRACT_RE.match(ln)
                if m:
                    beliefs_changed.append((m.group(1), "retract"))

        executed = 0
        executed_lines: list[str] = []
        if not dry_run:
            for ln in valid:
                try:
                    self._gs.execute(ln)
                    executed += 1
                    executed_lines.append(ln)
                except Exception as err:
                    rejected.append((ln, f"execute error: {err}"))
            # Scrape belief updates so the next ingest sees the current fact
            # state. Only lines that actually executed contribute - failed
            # ones leave the running state unchanged.
            _scrape_belief_updates(executed_lines, self._facts)

        duration_ms = int((time.perf_counter() - t0) * 1000)
        self._log_event(
            text, raw, valid, executed, rejected, entities_new,
            beliefs_changed, duration_ms, dry_run,
        )
        return IngestResult(
            statements=list(valid),
            executed=executed,
            rejected=rejected,
            entities_new=entities_new,
            beliefs_changed=beliefs_changed,
            duration_ms=duration_ms,
            raw_output=raw,
            skill_fingerprint=self._skill_fingerprint,
            dry_run=dry_run,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        """Fast char/4 estimate. llama-cpp-python's tokenize() is authoritative
        but costs a forward through the vocab table; for budget guards the
        cheap estimate is fine and conservative-enough (chars/4 over-counts
        for ASCII, under-counts for dense tokens - net neutral)."""
        return len(text) // 4 + 8

    def _log_event(
        self,
        input_text: str,
        raw: str,
        valid: list[str],
        executed: int,
        rejected: list[tuple[str, str]],
        entities_new: list[str],
        beliefs_changed: list[tuple[str, str]],
        duration_ms: int,
        dry_run: bool,
    ) -> None:
        _log.info(
            "bonsai.ingest: input_chars=%d raw_chars=%d stmts=%d exec=%d "
            "rejected=%d entities=%d beliefs=%d dur_ms=%d skill=%s dry_run=%s",
            len(input_text), len(raw), len(valid), executed, len(rejected),
            len(entities_new), len(beliefs_changed), duration_ms,
            self._skill_fingerprint, dry_run,
        )
