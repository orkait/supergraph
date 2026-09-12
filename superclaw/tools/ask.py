from __future__ import annotations

from typing import Any

from superclaw.tools import Permission, Result, Safety, SideEffect, Tool, ToolContext

NON_INTERACTIVE_MESSAGE = (
    "No interactive user is available to answer questions. "
    "Proceed with your best assumption, explicitly stating the assumptions you are making."
)


def parse_questions(args: dict[str, Any]) -> list[dict[str, Any]]:
    raw = args.get("questions")
    if not isinstance(raw, list) or not raw:
        raise ValueError("questions must be a non-empty array")
    out = []
    for i, q in enumerate(raw):
        if not isinstance(q, dict) or not str(q.get("question") or "").strip():
            raise ValueError(f"question {i + 1} needs a non-empty question")
        out.append(q)
    return out


class AskUser(Tool):
    name = "ask_user"
    deferred = True
    description = (
        "Ask the user one or more clarifying questions and wait for their answers. "
        "Only for decisions that are genuinely theirs to make; include 2-4 options and a recommended one when the answer is likely one of a small set."
    )
    parameters = {
        "type": "object",
        "properties": {
            "header": {"type": "string", "description": "Optional short heading shown above the questions."},
            "questions": {
                "type": "array",
                "description": "One or more questions to ask the user.",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question to ask."},
                        "header": {"type": "string", "description": "Optional 2-3 word tab label."},
                        "options": {"type": "array", "items": {"type": "string"}, "description": "Optional 2-4 suggested answers."},
                        "recommended": {"type": "string", "description": "Optional recommended option; must be one of options."},
                    },
                    "required": ["question"],
                },
            },
        },
        "required": ["questions"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.NONE, Permission.ALLOW, "Asks the user a question.")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        try:
            parse_questions(args)
        except ValueError as e:
            return Result.error(f"Error: invalid arguments for ask_user: {e}")
        return Result.success(NON_INTERACTIVE_MESSAGE)
