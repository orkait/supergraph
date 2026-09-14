from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from superclaw.runtime import Message
from superclaw.settings import LIMITS
from superclaw.text import oneline

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from superclaw.runtime import CancelCheck, Provider

_OBJECT_START: Final = re.compile(r"\{")
_PROMPTS: Final = Path(__file__).parent / "prompts"

HEADER: Final = (
    "An intent stage read the request before you and decomposed it. This is the contract for this turn, not a suggestion: "
    "the subgoals are what done means, and the open questions are reads to perform, not guesses to make."
)
SUBGOALS_TITLE: Final = "Subgoals, in order, each one independently checkable:"
QUERIES_TITLE: Final = "Open questions about this workspace; answer each by reading, searching or running something, never by guessing:"
SETTLED_TITLE: Final = "Settled by the user, treat as given:"
UNRESOLVED_TITLE: Final = "The user did not settle these; say which assumption you made rather than deciding silently:"


@dataclass(frozen=True, slots=True)
class Intent:
    goal: str = ""
    subgoals: tuple[str, ...] = ()
    queries: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()
    answered: tuple[tuple[str, str], ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.goal or self.subgoals or self.queries or self.unknowns or self.answered)

    @property
    def blocked(self) -> bool:
        return bool(self.unknowns)

    def settled(self, answers: Sequence[str]) -> Intent:
        replies = tuple(zip(self.unknowns, answers, strict=False))
        kept = tuple((question, oneline(answer)) for question, answer in replies if answer.strip())
        unresolved = tuple(question for question in self.unknowns if question not in {q for q, _ in kept})
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
                _listed(UNRESOLVED_TITLE, self.unknowns),
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
    return (_PROMPTS / "decompose.md").read_text().strip()


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    cleaned = [oneline(item) for item in value if isinstance(item, str) and item.strip()]
    return tuple(cleaned[: LIMITS.intent_items_max])


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
            _strings(raw.get("unknowns")),
        )
        if not parsed.empty:
            return parsed
    return Intent()


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
