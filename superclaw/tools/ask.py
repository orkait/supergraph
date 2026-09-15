from __future__ import annotations

from typing import Any

from superclaw.settings import LIMITS
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
        options = [str(o).strip() for o in (q.get("options") or []) if str(o).strip()]
        if not LIMITS.ask_options_min <= len(options) <= LIMITS.ask_options_max:
            raise ValueError(f"question {i + 1} needs {LIMITS.ask_options_min} to {LIMITS.ask_options_max} options; "
                             "the user can still type an answer of their own, so offer the likely ones rather than asking an open question")
        recommended = str(q.get("recommended") or "").strip()
        if recommended and recommended not in options:
            raise ValueError(f"question {i + 1}: recommended must be one of its options")
        out.append({**q, "options": options, "recommended": recommended or options[0]})
    return out


class AskUser(Tool):
    name = "ask_user"
    deferred = True
    description = (
        "Ask the user one or more clarifying questions and wait for their answers. Only for decisions that are genuinely theirs to make. "
        "Never ask an open question: every question carries 3 to 5 concrete options with one marked recommended. "
        "The user can always type something else, so options are your best reading of the likely answers, not a limit on theirs."
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
                        "options": {"type": "array", "items": {"type": "string"}, "description": "The likely answers, as concrete choices.",
                                    "minItems": LIMITS.ask_options_min, "maxItems": LIMITS.ask_options_max},
                        "recommended": {"type": "string", "description": "The option you recommend; must be one of options."},
                    },
                    "required": ["question", "options", "recommended"],
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
