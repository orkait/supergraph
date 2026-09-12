from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from superclaw.compaction import render_transcript
from superclaw.runtime import Message, Provider
from superclaw.tools.plan import format_plan

TRANSCRIPT_CLAMP = 24_000
_JSON = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Verdict:
    passed: bool
    reason: str
    next_action: str


def verifier_prompt() -> str:
    return (Path(__file__).parent / "prompts" / "verifier.md").read_text().strip()


def parse_verdict(text: str) -> Verdict:
    match = _JSON.search(text or "")
    if not match:
        return Verdict(False, "verifier returned no JSON verdict", "continue the task and finish with a verifiable result")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Verdict(False, "verifier returned malformed JSON", "continue the task and finish with a verifiable result")
    return Verdict(bool(data.get("passed")), str(data.get("reason") or ""), str(data.get("nextAction") or ""))


def verify(provider: Provider, objective: str, messages: list[Message], plan: list[dict[str, str]]) -> Verdict:
    transcript = render_transcript([m for m in messages if m.role != "system"])
    if len(transcript) > TRANSCRIPT_CLAMP:
        transcript = transcript[-TRANSCRIPT_CLAMP:]
    body = f"Objective:\n{objective}\n\n{format_plan(plan)}\n\nTranscript (most recent last):\n{transcript}"
    request = [Message(role="system", content=verifier_prompt()), Message(role="user", content=body)]
    return parse_verdict(provider.complete(request, []).text)
