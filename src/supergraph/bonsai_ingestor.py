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


class BonsaiError(Exception):
    pass


class IngestEmpty(BonsaiError):
    pass


class IngestOverflow(BonsaiError):
    pass


@dataclass
class FactState:

    fact_id: str
    kind: str = ""
    value: str = ""
    confidence: float = 1.0
    source: str = ""
    retracted: bool = False
    retract_reason: str = ""


@dataclass
class IngestResult:

    statements: list[str] = field(default_factory=list)
    executed: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)
    entities_new: list[str] = field(default_factory=list)
    beliefs_changed: list[tuple[str, str]] = field(default_factory=list)
    duration_ms: int = 0
    raw_output: str = ""
    skill_fingerprint: str = ""
    dry_run: bool = False


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
    return _THINK_RE.sub("", raw).strip()


def _split_lines(cleaned: str) -> list[str]:
    out: list[str] = []
    for raw_ln in cleaned.splitlines():
        ln = raw_ln.strip()
        if not ln or _FENCE_RE.match(ln):
            continue
        out.append(ln)
    return out


def _dedupe_upserts(stmts: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
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


@dataclass
class ParsedTurn:

    entities: list[tuple[str, str]] = field(default_factory=list)
    entity_edges: list[tuple[str, str, str]] = field(default_factory=list)
    beliefs: list[tuple[str, str]] = field(default_factory=list)
    retracts: list[str] = field(default_factory=list)
    statements: list[str] = field(default_factory=list)


def _dsl_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _h_upsert(turn: ParsedTurn, ln: str) -> None:
    parts = ln.split(None, 2)
    if len(parts) < 3:
        return
    slug = parts[1].removeprefix("ent:").strip('"')
    name = parts[2].strip().strip('"')
    if slug and name:
        turn.entities.append((slug, name))


def _h_fact(turn: ParsedTurn, ln: str) -> None:
    parts = ln.split(None, 2)
    if len(parts) < 3:
        return
    topic = parts[1].removeprefix("fact:").strip('"')
    value = parts[2].strip().strip('"')
    if topic and value:
        turn.beliefs.append((f"fact:{topic}", value))


def _h_drop(turn: ParsedTurn, ln: str) -> None:
    parts = ln.split()
    if len(parts) < 2:
        return
    topic = parts[1].removeprefix("fact:").strip('"')
    if topic:
        turn.retracts.append(f"fact:{topic}")


def _h_edge(turn: ParsedTurn, ln: str) -> None:
    parts = ln.split()
    if len(parts) < 4:
        return
    from_id = parts[1].strip('"')
    to_id = parts[2].strip('"')
    kind = parts[3].strip('"')
    if not (from_id and to_id and kind):
        return
    from_slug = from_id.removeprefix("ent:")
    to_slug = to_id.removeprefix("ent:")
    turn.entity_edges.append((from_slug, to_slug, kind))


def _h_query(template: str):
    def h(turn: ParsedTurn, ln: str) -> None:
        parts = ln.split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            return
        q = _dsl_escape(parts[1].strip())
        turn.statements.append(template.replace("{q}", q))
    return h


def _h_walk(template: str):
    def h(turn: ParsedTurn, ln: str) -> None:
        parts = ln.split()
        if len(parts) < 2:
            return
        anchor = parts[1].strip('"')
        if anchor:
            turn.statements.append(template.replace("{id}", _dsl_escape(anchor)))
    return h


def _h_plain(template: str):
    def h(turn: ParsedTurn, _ln: str) -> None:
        turn.statements.append(template)
    return h


def _h_snapshot(turn: ParsedTurn, ln: str) -> None:
    from datetime import datetime, timezone
    parts = ln.split(None, 1)
    if len(parts) >= 2 and parts[1].strip():
        name = parts[1].strip().strip('"')
    else:
        name = datetime.now(timezone.utc).strftime("snap-%Y%m%dT%H%M%SZ")
    turn.statements.append(f'SYS SNAPSHOT "{_dsl_escape(name)}"')


def _h_slug(template: str):
    def h(turn: ParsedTurn, ln: str) -> None:
        parts = ln.split()
        if len(parts) < 2:
            return
        slug = parts[1].removeprefix("ent:").strip('"')
        if slug:
            turn.statements.append(template.replace("{slug}", _dsl_escape(slug)))
    return h


def _h_topic(template: str):
    def h(turn: ParsedTurn, ln: str) -> None:
        parts = ln.split()
        if len(parts) < 2:
            return
        topic = parts[1].removeprefix("fact:").strip('"')
        if topic:
            turn.statements.append(template.replace("{topic}", _dsl_escape(topic)))
    return h


def _h_pair(template: str):
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
    def h(turn: ParsedTurn, ln: str) -> None:
        parts = ln.split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            return
        turn.statements.append(template.replace("{body}", parts[1].strip()))
    return h


def _h_update_edge(turn: ParsedTurn, ln: str) -> None:
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
    parts = ln.split()
    valid = {"COMPACT", "STRINGS", "EDGES", "VECTORS", "BLOBS", "CACHE"}
    if len(parts) >= 2:
        t = parts[1].strip().upper()
        if t in valid:
            turn.statements.append(f'SYS OPTIMIZE {t}')
    else:
        turn.statements.append('SYS OPTIMIZE')


def _h_clear(turn: ParsedTurn, ln: str) -> None:
    parts = ln.split()
    if len(parts) < 2:
        return
    t = parts[1].strip().upper()
    if t in ("LOG", "CACHE"):
        turn.statements.append(f'SYS CLEAR {t}')


def _h_wal(turn: ParsedTurn, ln: str) -> None:
    parts = ln.split()
    if len(parts) < 2:
        return
    a = parts[1].strip().upper()
    if a in ("STATUS", "REPLAY"):
        turn.statements.append(f'SYS WAL {a}')


def _h_vault_triplet(template: str):
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
    parts = ln.split(None, 1)
    if len(parts) >= 2 and parts[1].strip():
        turn.statements.append(f'NODES WHERE {parts[1].strip()} LIMIT 20')
    else:
        turn.statements.append('NODES LIMIT 20')


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
    "UPSERT": _h_upsert,
    "BELIEF": _h_fact, "BELIEVE": _h_fact, "ASSERT": _h_fact, "FACT": _h_fact,
    "RETRACT": _h_drop, "DROP": _h_drop,

    "EDGE": _h_edge, "CREATE_EDGE": _h_edge,
    "UPDATE_EDGE": _h_update_edge,
    "DELETE_EDGE": _H_EX,
    "DELETE_EDGES_FROM": _H_DEF,
    "DELETE_EDGES_TO": _H_DET,
    "EDGES_FROM": _H_EF,
    "EDGES_TO": _H_ET,

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

    "UPDATE_NODES": _H_UNS,
    "DELETE_NODES": _H_DNS,

    "REMEMBER": _H_RM,
    "SIMILAR": _H_SM, "SIMILAR_TO": _H_SM,
    "LEXICAL": _H_LX, "LEXICAL_SEARCH": _H_LX, "SEARCH": _H_LX,
    "ANSWER": _H_AQ,

    "COUNT_NODES": _H_PLAIN_COUNT_NODES,
    "COUNT_EDGES": _H_PLAIN_COUNT_EDGES,

    "RECALL": _H_RL,
    "TRAVERSE": _H_TR,
    "ANCESTORS": _H_AN,
    "DESCENDANTS": _H_DE,
    "SUBGRAPH": _H_SG,

    "PATH": _H_PA,
    "PATHS": _H_PAS,
    "SHORTEST": _H_SP, "SHORTEST_PATH": _H_SP,
    "DISTANCE": _H_DI,
    "WEIGHTED_SHORTEST": _H_WSP, "WEIGHTED_SHORTEST_PATH": _H_WSP,
    "WEIGHTED_DISTANCE": _H_WDI,
    "COMMON": _H_CO, "COMMON_NEIGHBORS": _H_CO,

    "MATCH": _H_MA,
    "AGGREGATE": _H_AG,

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

    "BIND_CONTEXT": _H_BC,
    "DISCARD_CONTEXT": _H_XC,
    "INGEST": _H_IG,

    "SNAPSHOT": _h_snapshot,
    "ROLLBACK": _H_SR,
    "SNAPSHOTS": _H_PLAIN_SNAPSHOTS,
    "COMPACT": _H_PLAIN_COMPACT,
    "OPTIMIZE": _h_optimize,

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

    "REGISTER_NODE": _H_SRN, "REGISTER_NODE_KIND": _H_SRN,
    "REGISTER_EDGE": _H_SRE, "REGISTER_EDGE_KIND": _H_SRE,
    "UNREGISTER": _h_unregister,

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

    "CRON_ADD": _h_cron_add,
    "CRON_DELETE": _H_Q_CRON_DELETE,
    "CRON_ENABLE": _H_Q_CRON_ENABLE,
    "CRON_DISABLE": _H_Q_CRON_DISABLE,
    "CRON_LIST": _H_PLAIN_CRON_LIST,
    "CRON_RUN": _H_Q_CRON_RUN,

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
    from supergraph.entity_resolver import (
        EDGE_REFERS_TO, KIND_ENTITY, KIND_MENTION,
        make_entity_id, make_mention_id,
        resolve_and_create_entity,
    )

    out: list[str] = []
    text_esc = _dsl_escape(text)
    session_esc = _dsl_escape(session_id)
    role_esc = _dsl_escape(role)
    msg_esc = _dsl_escape(msg_id)

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
    def _sub(match: re.Match) -> str:
        ref = match.group(1)
        bare = ref[4:] if ref.startswith("ent:") else ref
        eid = slug_to_entity.get(bare)
        if not eid and graph_slugs:
            eid = graph_slugs.get(bare)
        return f'"{eid}"' if eid else f'"ent:{bare}"'
    return _ENT_FROM_ID_RE.sub(_sub, stmt)


def _render_known_facts_block(facts: dict[str, FactState], max_facts: int = 40) -> str:
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


_DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "bonsai_dsl_prompt.txt"
_DEFAULT_LITE_PROMPT_PATH = Path(__file__).resolve().parent / "bonsai_dsl_prompt_lite.txt"


class BonsaiIngestor:

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
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._chat_format = chat_format
        self._n_threads = n_threads
        self._n_batch = n_batch
        self._flash_attn = flash_attn
        self._n_gpu_layers = n_gpu_layers

        self._skill_text = ""
        self._skill_fingerprint = ""
        self._system_prompt = ""
        self._reload_skill()

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

        self._facts: dict[str, FactState] = {}

        self._kv_cache_path = Path(kv_cache_path) if kv_cache_path else None

        self._ner_model_dir = Path(ner_model_dir) if ner_model_dir else None
        self._ner_score_threshold = ner_score_threshold
        self._ner_max_hints = max(0, int(ner_max_hints))
        self._ner_disabled = False


    def _reload_skill(self) -> None:
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
        return {
            "model_path": str(self._model_path),
            "model_size_bytes": self._model_path.stat().st_size,
            "skill_fingerprint": self._skill_fingerprint,
            "n_ctx": self._n_ctx,
            "chat_format": self._chat_format,
        }

    def _try_load_kv_cache(self, llm: Any) -> bool:
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
        with self._lock:
            self._llm = None


    @property
    def skill_fingerprint(self) -> str:
        return self._skill_fingerprint

    @property
    def facts(self) -> dict[str, FactState]:
        return dict(self._facts)

    def reset_facts(self) -> None:
        with self._lock:
            self._facts.clear()


    def warmup(self) -> None:
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

        assert msg_id is not None
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


    def _estimate_tokens(self, text: str) -> int:
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
