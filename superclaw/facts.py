from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from supergraph.core.errors import SuperGraphError

from superclaw.dsl import MS_PER_SECOND, age, edge, lit, now_ms, rows
from superclaw.redaction import redact
from superclaw.session import NAMESPACE
from superclaw.settings import FACT_KIND, FROM_EDGE, LEARNED_EDGE, LIMITS, SUPERSEDES_EDGE

FACT_LINE = re.compile(r"^\s*[-*]\s*(?P<text>[^|]+?)\s*(?:\|\s*(?P<quote>.+?))?\s*$")
FACTS_HEADING = "Facts:"


def as_of_ms(text: str) -> int | None:
    try:
        when = datetime.fromisoformat(text.strip())
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return int(when.timestamp() * MS_PER_SECOND)


@dataclass(frozen=True)
class Fact:
    id: str
    text: str
    source: str
    observed_at: int
    confidence: float

    def line(self) -> str:
        return f"- ({age(self.observed_at)}, {self.source}) {self.text}"


def ident(text: str) -> str:
    return f"{FACT_KIND}:" + hashlib.sha1(" ".join(text.split()).lower().encode("utf-8")).hexdigest()[: LIMITS.id_hash_chars]


def parse(text: str) -> list[tuple[str, str]]:
    head, sep, tail = text.partition(FACTS_HEADING)
    if not sep:
        return []
    found: list[tuple[str, str]] = []
    for line in tail.splitlines():
        match = FACT_LINE.match(line)
        if match and match.group("text").strip():
            found.append((" ".join(match.group("text").split()), " ".join((match.group("quote") or "").split())))
    return found[: LIMITS.facts_per_page_max]


class Facts:
    def __init__(self, gs: Any) -> None:
        self._gs = gs

    def _x(self, query: str) -> Any:
        return self._gs.execute(query, namespace=NAMESPACE)

    def assert_(self, text: str, source: str, *, session_id: str = "", confidence: float = LIMITS.web_fact_confidence,
                observed_at: int | None = None, quote: str = "", page_ref: str = "") -> Fact:
        text = " ".join(text.split())
        if not text or not source:
            raise ValueError("a fact needs text and a source")
        if redact(text)[1] or redact(quote)[1]:
            raise ValueError("refused: the fact contains a secret")
        at = observed_at or now_ms()
        node = ident(text)
        fields = f'kind = {lit(FACT_KIND)} source = {lit(source)} observed_at = {at} confidence = {confidence} sid = {lit(session_id)} quote = {lit(quote)}'
        try:
            self._x(f"CREATE NODE {lit(node)} {fields} DOCUMENT {lit(text)}")
        except SuperGraphError as e:
            if "exist" not in str(e).lower():
                raise
            self._x(f"UPDATE NODE {lit(node)} SET observed_at = {at} confidence = {confidence} source = {lit(source)}")
        self._x(f"ASSERT {lit(node)} {fields} CONFIDENCE {confidence} SOURCE {lit(source)} EVENT_AT {at}")
        if page_ref:
            self.link(node, f"obs:{page_ref}", FROM_EDGE)
        if session_id:
            self.link(node, f"session:{session_id}", LEARNED_EDGE)
        return Fact(node, text, source, at, confidence)

    def link(self, source: str, target: str, kind: str) -> bool:
        return edge(self._gs, source, target, kind, NAMESPACE)

    def retract(self, node: str, reason: str) -> None:
        self._x(f"RETRACT {lit(node)} REASON {lit(reason)}")

    def supersede(self, old: str, text: str, source: str, **kw: Any) -> Fact:
        fact = self.assert_(text, source, **kw)
        if fact.id != old:
            self.link(fact.id, old, SUPERSEDES_EDGE)
            self.retract(old, f"superseded by {fact.id}")
        return fact

    def load(self, node: str) -> Fact | None:
        try:
            data = self._x(f"NODE {lit(node)} WITH DOCUMENT").data or {}
        except SuperGraphError:
            return None
        text = (data.get("_document") or "").strip()
        if not text:
            return None
        return Fact(node, text, str(data.get("source") or ""), int(data.get("observed_at") or 0), float(data.get("confidence") or 0))

    def search(self, query: str, limit: int = LIMITS.fact_recall_limit, as_of: int | None = None) -> list[Fact]:
        window = f" AND observed_at <= {int(as_of)}" if as_of else ""
        try:
            found = rows(self._x(f"REMEMBER {lit(query)} LIMIT {int(limit)} WHERE kind = {lit(FACT_KIND)}{window}"))
        except SuperGraphError:
            return []
        return [fact for row in found if (fact := self.load(str(row["id"]))) is not None]

    def recent(self, limit: int = LIMITS.fact_list_limit, as_of: int | None = None) -> list[Fact]:
        window = f" AND observed_at <= {int(as_of)}" if as_of else ""
        found = rows(self._x(f"NODES WHERE kind = {lit(FACT_KIND)}{window} ORDER BY observed_at DESC LIMIT {int(limit)}"))
        return [fact for row in found if (fact := self.load(str(row["id"]))) is not None]

    @staticmethod
    def render(facts: list[Fact]) -> str:
        return "\n".join(fact.line() for fact in facts)
