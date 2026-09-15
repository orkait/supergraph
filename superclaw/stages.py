from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from superclaw.runtime import Message
from superclaw.settings import LIMITS, PROMPTS_DIR, RECOMMENDED_MARK
from superclaw.text import oneline

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from superclaw.runtime import CancelCheck, Provider

_OBJECT_START: Final = re.compile(r"\{")

HEADER: Final = (
    "An intent stage read the request before you and decomposed it. This is the contract for this turn, not a suggestion: "
    "the subgoals are what done means, and the open questions are reads to perform, not guesses to make."
)
SUBGOALS_TITLE: Final = "Subgoals, in order, each one independently checkable:"
QUERIES_TITLE: Final = "Open questions about this workspace; answer each by reading, searching or running something, never by guessing:"
SETTLED_TITLE: Final = "Settled by the user, treat as given:"
ANSWER_PREFIX: Final = "Answer by reading or running something:"
SETTLED_PREFIX: Final = "Settled by the user:"
SETTLED_SEP: Final = " -> "
UNRESOLVED_TITLE: Final = "The user did not settle these; say which assumption you made rather than deciding silently:"


@dataclass(frozen=True, slots=True)
class Unknown:
    question: str
    options: tuple[str, ...] = ()
    recommended: str = ""

    def asked(self) -> dict[str, object]:
        return {"question": self.question, "options": list(self.options), "recommended": self.recommended}

    def line(self) -> str:
        shown = ", ".join(f"{o} {RECOMMENDED_MARK}" if o == self.recommended else o for o in self.options)
        return f"{self.question} [{shown}]" if shown else self.question


@dataclass(frozen=True, slots=True)
class Intent:
    goal: str = ""
    subgoals: tuple[str, ...] = ()
    queries: tuple[str, ...] = ()
    unknowns: tuple[Unknown, ...] = ()
    answered: tuple[tuple[str, str], ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.goal or self.subgoals or self.queries or self.unknowns or self.answered)

    @property
    def blocked(self) -> bool:
        return bool(self.unknowns)

    def plan_seed(self) -> tuple[str, ...]:
        return tuple(f"{ANSWER_PREFIX} {query}" for query in self.queries) + self.subgoals

    def remembered(self, known: Sequence[tuple[str, str]]) -> Intent:
        if not known:
            return self
        answers = dict(known)
        kept = tuple((question, answers[question]) for question in (u.question for u in self.unknowns) if question in answers)
        unresolved = tuple(unknown for unknown in self.unknowns if unknown.question not in answers)
        return Intent(self.goal, self.subgoals, self.queries, unresolved, (*self.answered, *kept))

    def settled(self, answers: Sequence[str]) -> Intent:
        replies = tuple(zip(self.unknowns, answers, strict=False))
        kept = tuple((unknown.question, oneline(answer)) for unknown, answer in replies if answer.strip())
        settled = {question for question, _ in kept}
        unresolved = tuple(unknown for unknown in self.unknowns if unknown.question not in settled)
        return Intent(self.goal, self.subgoals, self.queries, unresolved, (*self.answered, *kept))

    def block(self) -> str:
        if self.empty:
            return ""
        body = "\n".join(
            line
            for line in (
                HEADER,
                f"Goal: {self.goal}" if self.goal else "",
                _listed(SUBGOALS_TITLE, self.subgoals),
                _listed(QUERIES_TITLE, self.queries),
                _listed(SETTLED_TITLE, tuple(f"{question} {answer}" for question, answer in self.answered)),
                _listed(UNRESOLVED_TITLE, tuple(unknown.line() for unknown in self.unknowns)),
            )
            if line
        )
        return f"""<intent>
{body}
</intent>"""


def _listed(title: str, items: tuple[str, ...]) -> str:
    if not items:
        return ""
    bullets = "\n".join(f"- {item}" for item in items)
    return f"""{title}
{bullets}"""


def decompose_prompt() -> str:
    return (PROMPTS_DIR / "decompose.md").read_text().strip()


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    cleaned = [oneline(item) for item in value if isinstance(item, str) and item.strip()]
    return tuple(cleaned[: LIMITS.intent_items_max])


def _unknowns(value: object) -> tuple[Unknown, ...]:
    if not isinstance(value, list):
        return ()
    found: list[Unknown] = []
    for item in value[: LIMITS.intent_items_max]:
        if isinstance(item, str) and item.strip():
            found.append(Unknown(oneline(item)))
        elif isinstance(item, dict) and isinstance(question := item.get("question"), str) and question.strip():
            options = _strings(item.get("options"))[: LIMITS.ask_options_max]
            offered = item.get("recommended")
            chosen = oneline(offered) if isinstance(offered, str) else ""
            found.append(Unknown(oneline(question), options, chosen if chosen in options else next(iter(options), "")))
    return tuple(found)


def _objects(text: str) -> Iterator[dict[str, object]]:
    decoder = json.JSONDecoder()
    for match in _OBJECT_START.finditer(text):
        try:
            found, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(found, dict):
            yield found


def parse_intent(text: str) -> Intent:
    for raw in _objects(text or ""):
        goal = raw.get("goal")
        parsed = Intent(
            oneline(goal)[: LIMITS.intent_goal_chars] if isinstance(goal, str) else "",
            _strings(raw.get("subgoals")),
            _strings(raw.get("queries")),
            _unknowns(raw.get("unknowns")),
        )
        if not parsed.empty:
            return parsed
    return Intent()


def settled_note(question: str, answer: str) -> str:
    return f"{SETTLED_PREFIX} {oneline(question)}{SETTLED_SEP}{oneline(answer)}"


def settled_answer(note: str) -> str:
    asked, sep, given = note.partition(SETTLED_SEP)
    return oneline(given) if sep and SETTLED_PREFIX in asked else ""


def recall_settled(notes: Iterable[str], unknowns: tuple[Unknown, ...]) -> tuple[tuple[str, str], ...]:
    known = [note for note in notes if settled_answer(note)]
    found: list[tuple[str, str]] = []
    for unknown in unknowns:
        asked = oneline(unknown.question).lower()
        hit = next((note for note in known if asked and asked in note.lower()), "")
        if hit:
            found.append((unknown.question, settled_answer(hit)))
    return tuple(found)


def decompose(provider: Provider, request: str, cancelled: CancelCheck | None = None) -> Intent:
    asked = request.strip()
    if len(asked) < LIMITS.intent_min_chars:
        return Intent()
    reply = provider.complete(
        [Message(role="system", content=decompose_prompt()), Message(role="user", content=asked)],
        [],
        cancelled=cancelled,
    )
    return parse_intent(reply.text)
