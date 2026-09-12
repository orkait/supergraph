from __future__ import annotations

import re
from dataclasses import dataclass

from superclaw.settings import LIMITS

PLAN_TOOL = "update_plan"

_PROMISES = ("i'll ", "i will ", "let me ", "next, ", "next steps", "next step", "now i'll ", "now let me ", "remaining:", "todo:")

DROPPED_TOOL_CALL_NOTICE = (
    "Your previous tool call was malformed (it was missing a tool name or had invalid JSON arguments) and was not executed. "
    "Re-issue the tool call with a valid tool name and JSON arguments, or reply with your final answer."
)
EMPTY_TURN_NUDGE = (
    "Your previous response had no visible output and no tool calls. "
    "Continue the task by using a tool or reply with your final answer."
)
MAX_TURNS_FINAL_ANSWER_PROMPT = (
    "You have reached the tool-turn limit. Do not call tools. Give a concise final answer now: "
    "summarize what you completed, what you found, and any remaining blockers."
)

_CUES = (
    "let me ", "let's ", "now i'll ", "now i will ", "now let me ", "let me now ",
    "i'll now ", "i will now ", "next i'll ", "next, i'll ", "first, i'll ", "first let me ",
)


def continue_nudge(reason: str) -> str:
    return (
        f"You stopped without calling a tool, but the task is not finished ({reason}). "
        "Do not stop here: take the next concrete action with a tool now. "
        "If you are genuinely finished, first mark the plan complete with update_plan, then give your final summary."
    )


def tool_failure_hint(tool_name: str, schema_json: str, err_output: str) -> str:
    return (
        f"Your calls to the `{tool_name}` tool kept failing with the same error:\n{err_output.strip()}\n\n"
        f"The `{tool_name}` tool expects arguments matching this schema; match it exactly:\n{schema_json.strip()}\n\n"
        "Fix the arguments and try once more, or take a different approach."
    )


def tool_failure_stop_answer(tool_name: str, count: int) -> str:
    return (
        f"Agent stopped: the `{tool_name}` tool failed {count} times in a row with the same error, "
        "so I halted instead of looping further. Please check the request or adjust the tool arguments."
    )


def no_output_stop_answer(turns: int) -> str:
    return f"Agent stopped: {turns} turns produced no visible output and no tool calls, so I halted rather than continue silently."


def plan_stale_reminder(calls_since_update: int) -> str:
    return (
        f"Reminder: you've made {calls_since_update} tool calls but have not updated the plan in a while. "
        "Update the plan to reflect completed and remaining steps, then continue."
    )


def tool_only_progress_reminder(turns: int) -> str:
    return (
        f"Reminder: you've made {turns} consecutive tool-only turns without visible progress. "
        "Before calling more tools, summarize what you already know, state the next concrete step, and finish if you have enough information."
    )


def ends_with_continuation_cue(text: str) -> bool:
    trimmed = text.strip()
    if not trimmed:
        return False
    last = next((line.strip().lower() for line in reversed(trimmed.split("\n")) if line.strip()), "")
    if not last.endswith(":"):
        return False
    clause = last[last.rfind(". ") + 2:] if ". " in last else last
    if clause.startswith("let me know"):
        return False
    return any(clause.startswith(cue) for cue in _CUES)


def ends_with_promise(text: str) -> bool:
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    if not paragraphs:
        return False
    last = paragraphs[-1].lower()
    first_line = last.split("\n", 1)[0]
    if first_line.startswith("let me know"):
        return False
    return any(first_line.startswith(p) for p in _PROMISES) or ends_with_continuation_cue(last)


def promise_nudge() -> str:
    return (
        "Your last paragraph describes work still to do. Do that work now with tool calls instead of ending the turn; "
        "if it is genuinely finished, state the result without a next-steps list."
    )


def identical_call_reminder(name: str, count: int) -> str:
    return (
        f"You have called `{name}` {count} times with identical input. Repeating the same call returns the same result; "
        "change the arguments, use a different tool, or state what you found."
    )


def calls_per_turn_reminder(count: int) -> str:
    return (
        f"This turn issued {count} tool calls. Stop and summarise what you have learned before issuing more; "
        "batch only independent read-only lookups."
    )


def error_signature(output: str) -> str:
    sig = output.strip().lower()
    sig = re.sub(r"(?:/[^\s/]+)+", "<path>", sig)
    sig = re.sub(r"\d+", "#", sig)
    return sig[:LIMITS.error_signature_chars]


@dataclass
class FailureOutcome:
    count: int
    hint: bool
    stop: bool


class Guards:
    def __init__(
        self,
        *,
        max_empty_turns: int = LIMITS.max_empty_turns,
        hint_at: int = LIMITS.failure_hint_at,
        stop_at: int = LIMITS.failure_stop_at,
        stale_tool_calls: int = LIMITS.stale_plan_tool_calls,
        tool_only_at: int = LIMITS.tool_only_reminder_at,
    ) -> None:
        self.max_empty_turns = max_empty_turns
        self.hint_at = hint_at
        self.stop_at = stop_at
        self.stale_tool_calls = stale_tool_calls
        self.tool_only_at = tool_only_at
        self._empty = 0
        self._silent = 0
        self._tool_only_reminded = False
        self._failure: tuple[str, str] | None = None
        self._failure_count = 0
        self._calls_since_plan = 0
        self._plan_reminded = False
        self._identical: tuple[str, str] | None = None
        self._identical_count = 0

    def observe_identical(self, name: str, arguments: str) -> str | None:
        key = (name, arguments)
        self._identical_count = self._identical_count + 1 if key == self._identical else 1
        self._identical = key
        return identical_call_reminder(name, self._identical_count) if self._identical_count == LIMITS.identical_call_at else None

    def observe_turn(self, text: str, tool_calls: int) -> bool:
        if not text.strip() and tool_calls == 0:
            self._empty += 1
        else:
            self._empty = 0
        return self._empty >= self.max_empty_turns

    def observe_tool_result(self, name: str, failed: bool, output: str) -> FailureOutcome:
        if not failed:
            self._failure = None
            self._failure_count = 0
            return FailureOutcome(0, False, False)
        key = (name, error_signature(output))
        if key == self._failure:
            self._failure_count += 1
        else:
            self._failure = key
            self._failure_count = 1
        return FailureOutcome(
            self._failure_count,
            self._failure_count == self.hint_at,
            self._failure_count >= self.stop_at,
        )

    def observe_tool_call(self, name: str) -> None:
        if name == PLAN_TOOL:
            self._calls_since_plan = 0
            self._plan_reminded = False
        else:
            self._calls_since_plan += 1

    def stale_plan_reminder(self, pending: bool) -> str | None:
        if pending and not self._plan_reminded and self._calls_since_plan >= self.stale_tool_calls:
            self._plan_reminded = True
            return plan_stale_reminder(self._calls_since_plan)
        return None

    def tool_only_reminder(self, text: str, tool_calls: int) -> str | None:
        if tool_calls > 0 and not text.strip():
            self._silent += 1
        else:
            self._silent = 0
            self._tool_only_reminded = False
        if self._silent == self.tool_only_at and not self._tool_only_reminded:
            self._tool_only_reminded = True
            return tool_only_progress_reminder(self._silent)
        return None
