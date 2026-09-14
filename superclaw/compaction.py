from __future__ import annotations

import json
import re
from dataclasses import dataclass
from collections.abc import Callable

from superclaw.runtime import Message, message_tokens
from superclaw.settings import LIMITS

SUMMARY_LABEL = "[Summary of earlier conversation]"
RESUME_NOTE = "Continue from here. Do not acknowledge this summary or restart finished work; the last user message is the current request."
PRESERVED_LABEL = "## Preserved state (carried across compaction)"
TRANSCRIPT_NOTE = "The full transcript is session {sid}; `recall` restores any §ref named above in full."
_WORD = re.compile(r"\S+")

SUMMARY_INSTRUCTIONS = (
    "You are compacting a coding-assistant conversation to save context. Write a dense, factual summary in exactly these nine sections, "
    "each as a heading followed by terse bullets:\n"
    "1. Task and goals\n2. User messages, every one, quoted verbatim in order\n3. Decisions made and why\n"
    "4. Files created or modified, with paths\n5. Commands run and their important results\n6. Errors hit and how they were resolved\n"
    "7. Constraints and security rules stated by the user or the system, verbatim\n8. Open items and unresolved questions\n9. The next concrete step\n"
    "Do not invent details. If the brief begins with [previous summary], treat its facts as established and carry every one of them forward."
)


def summary_instructions(notes: list[str] | None = None) -> str:
    return SUMMARY_INSTRUCTIONS + ("\nAdditional instructions from the user's hooks:\n" + "\n".join(notes) if notes else "")


@dataclass
class CompactionResult:
    messages: list[Message]
    removed: int = 0
    preserved: int = 0
    summary: str = ""
    compacted: bool = False


PRUNE_MARKER = "\n\n[... tool result middle pruned to fit the context window{recall} ...]\n\n"
_REF = re.compile(r"§([0-9a-f]{8,})")


def _system_end(messages: list[Message]) -> int:
    end = 0
    while end < len(messages) and messages[end].role == "system":
        end += 1
    return end


def cut_point(messages: list[Message], keep_tokens: int = LIMITS.compaction_keep_tokens) -> int:
    system_end = _system_end(messages)
    boundary, kept = len(messages), 0
    while boundary > system_end:
        kept += message_tokens(messages[boundary - 1])
        if kept > keep_tokens:
            break
        boundary -= 1
    while system_end < boundary < len(messages) and messages[boundary].role == "tool":
        boundary -= 1
    if boundary in (system_end, len(messages)) and kept > keep_tokens:
        boundary = max((i for i in range(system_end, len(messages)) if messages[i].role == "assistant"), default=system_end)
    return boundary


def prune_tool_results(messages: list[Message], upto: int) -> list[tuple[int, str, str]]:
    pruned = []
    for index, m in enumerate(messages[:upto]):
        if m.role == "tool" and len(m.content) > LIMITS.prune_threshold_chars:
            match = _REF.search(m.content)
            ref = match.group(1) if match else ""
            marker = PRUNE_MARKER.format(recall=f"; recall §{ref} to expand" if ref else "")
            pruned.append((index, m.content[:LIMITS.prune_head_chars] + marker + m.content[-LIMITS.prune_tail_chars:], ref))
    return pruned


def _clamp(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"… [{len(text) - limit} more chars]"


def _words(text: str, budget: int) -> str:
    words = _WORD.findall(text)
    if len(words) <= budget:
        return text.strip()
    return " ".join(words[:budget]) + f" … [{len(words) - budget} more words]"


def _first_line(text: str) -> str:
    return text.strip().split("\n", 1)[0]


def _call_line(name: str, arguments: str) -> str:
    return f"* {name}({_clamp(arguments, LIMITS.compaction_arg_bytes)})"


def render_transcript(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        if m.role == "tool":
            lines.append(f"[tool {m.tool_call_id}{' error' if m.is_error else ''}] {_clamp(m.content, LIMITS.compaction_tool_result_clamp)}")
            continue
        if m.content:
            lines.append(f"[{m.role}] {m.content}")
        for c in m.tool_calls:
            lines.append(f"[{m.role} tool_call {c.id}] {c.name}({_clamp(c.arguments, LIMITS.compaction_tool_args_clamp)})")
    return "\n".join(lines)


def _tool_names(messages: list[Message]) -> dict[str, str]:
    return {c.id: c.name for m in messages for c in m.tool_calls}


def _assistant_section(index: int, m: Message) -> str:
    lines = []
    if m.content.strip():
        lines.append(_words(m.content, LIMITS.compaction_assistant_words))
    calls = m.tool_calls[-LIMITS.compaction_calls_per_turn:]
    if len(m.tool_calls) > len(calls):
        lines.append(f"* ({len(m.tool_calls) - len(calls)} earlier tool calls omitted)")
    lines += [_call_line(c.name, c.arguments) for c in calls]
    if not lines:
        return ""
    return f"[assistant #{index}]\n" + "\n".join(lines)


def _tool_section(index: int, m: Message, name: str) -> str:
    if m.is_error:
        return f"[tool_error #{index}] {name}\n{_clamp(_first_line(m.content), LIMITS.compaction_error_bytes)}"
    if name == "ask_user":
        return f"[user_answer #{index}]\n{_clamp(m.content, LIMITS.compaction_result_bytes)}"
    if name in ("write_file", "edit_file"):
        return f"[tool_result #{index}] {name}\n{_clamp(_first_line(m.content), LIMITS.compaction_error_bytes)}"
    return ""


def _fit(brief: str) -> str:
    if len(brief) <= LIMITS.compaction_brief_max_bytes:
        return brief
    marker = "\n\n...[middle omitted to fit the compaction budget]...\n\n"
    head = int((LIMITS.compaction_brief_max_bytes - len(marker)) * LIMITS.compaction_head_share)
    tail = LIMITS.compaction_brief_max_bytes - len(marker) - head
    return brief[:head] + marker + brief[-tail:]


def project(messages: list[Message]) -> str:
    names = _tool_names(messages)
    sections: list[str] = []
    previous = ""
    for index, m in enumerate(messages):
        if m.role == "user":
            content = m.content.split(PRESERVED_LABEL, 1)[0].strip()
            if content.startswith(SUMMARY_LABEL):
                previous = _clamp(content[len(SUMMARY_LABEL):].replace(RESUME_NOTE, "").strip(), LIMITS.compaction_previous_summary_bytes)
            elif content:
                sections.append(f"[user #{index}]\n{_words(content, LIMITS.compaction_user_words)}")
        elif m.role == "assistant":
            if section := _assistant_section(index, m):
                sections.append(section)
        elif m.role == "tool":
            if section := _tool_section(index, m, names.get(m.tool_call_id, "tool")):
                sections.append(section)
    brief = _fit("\n\n".join(sections))
    if previous:
        brief = f"[previous summary]\n{previous}\n\n{brief}"
    return brief.strip()


def preserved_state(middle: list[Message], plan_text: str) -> str:
    skills: list[str] = []
    edits: list[str] = []
    for m in middle:
        for c in m.tool_calls:
            try:
                args = json.loads(c.arguments or "{}")
            except ValueError:
                continue
            if c.name == "skill" and args.get("name"):
                skills.append(str(args["name"]))
            if c.name in ("write_file", "edit_file") and args.get("path"):
                edits.append(str(args["path"]))
    parts = []
    if plan_text:
        parts.append(plan_text)
    if skills:
        parts.append("Skills loaded: " + ", ".join(dict.fromkeys(skills)))
    if edits:
        parts.append("Files edited: " + ", ".join(list(dict.fromkeys(edits))[-20:]))
    return f"{PRESERVED_LABEL}\n" + "\n".join(parts) if parts else ""


def compact(
    messages: list[Message],
    *,
    keep_tokens: int = LIMITS.compaction_keep_tokens,
    summarize: Callable[[str], str],
    plan_text: str = "",
    footer: str = "",
) -> CompactionResult:
    system_end = _system_end(messages)
    boundary = cut_point(messages, keep_tokens)
    middle = messages[system_end:boundary]
    if not middle:
        return CompactionResult(messages=list(messages), preserved=len(messages))
    summary = summarize(project(middle)).strip()
    content = f"{SUMMARY_LABEL}\n{summary}"
    if state := preserved_state(middle, plan_text):
        content += f"\n\n{state}"
    content += f"\n\n{RESUME_NOTE}" + (f"\n{footer}" if footer else "")
    compacted = [*messages[:system_end], Message(role="user", content=content), *messages[boundary:]]
    return CompactionResult(messages=compacted, removed=len(middle), preserved=len(messages) - len(middle), summary=summary, compacted=True)
