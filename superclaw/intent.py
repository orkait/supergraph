from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path

from superclaw.runtime import Message, Provider

_JSON = re.compile(r"\{.*?\}", re.DOTALL)


class Kind(str, Enum):
    ANSWER = "answer"
    DIAGNOSE = "diagnose"
    CHANGE = "change"
    MONITOR = "monitor"


GUIDANCE = {
    Kind.ANSWER: "answer or explain; this does not authorize writes, shell mutations or external side effects",
    Kind.DIAGNOSE: "diagnose; determine the cause and explain it, do not implement the fix unless the user asks",
    Kind.CHANGE: "change; implement and verify in proportion to risk",
    Kind.MONITOR: "monitor; wait for the external state and report; unchanged state is expected and is not a blocker",
}


def intent_prompt() -> str:
    return (Path(__file__).parent / "prompts" / "intent.md").read_text().strip()


def parse_kind(text: str) -> Kind:
    match = _JSON.search(text or "")
    if match:
        try:
            value = str(json.loads(match.group(0)).get("kind") or "").strip().lower()
            if value in Kind._value2member_map_:
                return Kind(value)
        except (json.JSONDecodeError, AttributeError):
            pass
    return Kind.CHANGE


def classify(provider: Provider, request: str) -> Kind:
    completion = provider.complete([Message(role="system", content=intent_prompt()), Message(role="user", content=request)], [])
    return parse_kind(completion.text)
