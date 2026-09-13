from __future__ import annotations

import re
from typing import Any

from superclaw.runtime import Completion, Message, ToolCall, Usage, to_wire
from superclaw.settings import ERROR_HINTS, LIMITS

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def hint(message: str, tui: bool) -> str:
    low = message.lower()
    for entry in ERROR_HINTS:
        if any(needle in low for needle in entry.needles):
            return entry.tui if tui else entry.cli
    return ""


def _completion(**kwargs: Any) -> Any:
    import litellm

    litellm.suppress_debug_info = True
    litellm.drop_params = True
    return litellm.completion(**kwargs)


def parse_response(resp: Any) -> Completion:
    choice = resp.choices[0]
    message = choice.message
    text = _THINK.sub("", message.content or "").strip()
    calls: list[ToolCall] = []
    for tc in getattr(message, "tool_calls", None) or []:
        name = getattr(tc.function, "name", "") or ""
        if not name:
            continue
        calls.append(ToolCall(id=tc.id or f"call_{len(calls)}", name=name, arguments=tc.function.arguments or "{}"))
    usage = getattr(resp, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None)
    return Completion(
        text=text,
        tool_calls=calls,
        usage=Usage(
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cache_read_tokens=int(getattr(details, "cached_tokens", 0) or 0),
        ),
        finish_reason=getattr(choice, "finish_reason", "") or "",
    )


class LitellmProvider:
    def __init__(
        self,
        chain: list[dict[str, Any]],
        *,
        max_tokens: int = LIMITS.completion_max_tokens,
        temperature: float = 0.0,
        timeout_s: int = LIMITS.completion_timeout_s,
        effort: str = "",
    ) -> None:
        if not chain:
            raise ValueError("LitellmProvider needs at least one provider in the chain")
        self._chain = chain
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout_s = timeout_s
        self.effort = effort

    @property
    def model(self) -> str:
        return self._chain[0]["litellm_model"]

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> Completion:
        last_err: Exception | None = None
        for provider in self._chain:
            kwargs: dict[str, Any] = {
                "model": provider["litellm_model"],
                "messages": to_wire(messages),
                "api_key": provider.get("api_key"),
                "api_base": provider.get("api_base"),
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "timeout": self._timeout_s,
                "stream": False,
            }
            if provider.get("account_id"):
                kwargs["account_id"] = provider["account_id"]
            if self.effort:
                kwargs["reasoning_effort"] = self.effort
            if provider.get("extra_headers"):
                kwargs["extra_headers"] = provider["extra_headers"]
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            try:
                return parse_response(_completion(**kwargs))
            except Exception as e:
                last_err = e
        raise RuntimeError(f"all providers failed: {last_err}")
