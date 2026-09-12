from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from superclaw.runtime import Message

SUMMARY_LABEL = "[Summary of earlier conversation]"
TRIGGER_RATIO = 0.7
DEFAULT_PRESERVE_LAST = 6
TOOL_RESULT_CLAMP = 2000
TOOL_ARGS_CLAMP = 500

SUMMARY_INSTRUCTIONS = (
    "You are compacting a coding-assistant conversation to save context. "
    "Write a dense, factual summary of the conversation so far. Preserve: the user's goals and explicit constraints; "
    "decisions made and why; files created or modified (with paths) and key code changes; commands run and their important "
    "results; and anything still in progress or unresolved. Omit pleasantries. Use terse bullet points. Do not invent details. "
    "If the conversation already begins with an earlier summary block, treat its facts as established context and carry them "
    "forward into the new summary; never drop earlier information."
)


@dataclass
class CompactionResult:
    messages: list[Message]
    removed: int = 0
    preserved: int = 0
    summary: str = ""
    compacted: bool = False


def threshold(context_window: int) -> int:
    return int(context_window * TRIGGER_RATIO) if context_window > 0 else 0


def _clamp(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"… [{len(text) - limit} more chars]"


def render_transcript(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        if m.role == "tool":
            lines.append(f"[tool {m.tool_call_id}{' error' if m.is_error else ''}] {_clamp(m.content, TOOL_RESULT_CLAMP)}")
            continue
        if m.content:
            lines.append(f"[{m.role}] {m.content}")
        for c in m.tool_calls:
            lines.append(f"[{m.role} tool_call {c.id}] {c.name}({_clamp(c.arguments, TOOL_ARGS_CLAMP)})")
    return "\n".join(lines)


def compact(
    messages: list[Message],
    *,
    preserve_last: int = DEFAULT_PRESERVE_LAST,
    summarize: Callable[[list[Message]], str],
    preserved_state: str = "",
) -> CompactionResult:
    if preserve_last <= 0:
        preserve_last = DEFAULT_PRESERVE_LAST
    system_end = 0
    while system_end < len(messages) and messages[system_end].role == "system":
        system_end += 1
    boundary = max(len(messages) - preserve_last, system_end)
    while boundary > system_end and messages[boundary].role != "assistant":
        boundary -= 1
    middle = messages[system_end:boundary]
    if not middle:
        return CompactionResult(messages=list(messages), preserved=len(messages))
    summary = summarize(middle).strip()
    content = f"{SUMMARY_LABEL}\n{summary}"
    if preserved_state:
        content += f"\n\n{preserved_state}"
    compacted = [*messages[:system_end], Message(role="user", content=content), *messages[boundary:]]
    return CompactionResult(
        messages=compacted,
        removed=len(middle),
        preserved=len(messages) - len(middle),
        summary=summary,
        compacted=True,
    )
