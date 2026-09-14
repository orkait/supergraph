from __future__ import annotations

import re
from collections.abc import Callable
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


def _visible(text: str) -> str:
    cleaned = _THINK.sub("", text)
    start = cleaned.find("<think>")
    return cleaned if start < 0 else cleaned[:start]


def collect(chunks: Any, on_text: Callable[[str], None]) -> Completion:
    parts: list[str] = []
    calls: dict[Any, ToolCall] = {}
    order: list[Any] = []
    shown, finish, usage = "", "", Usage()
    for chunk in chunks:
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            details = getattr(chunk_usage, "prompt_tokens_details", None)
            usage = Usage(
                input_tokens=int(getattr(chunk_usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(chunk_usage, "completion_tokens", 0) or 0),
                cache_read_tokens=int(getattr(details, "cached_tokens", 0) or 0),
            )
        for choice in getattr(chunk, "choices", None) or []:
            finish = getattr(choice, "finish_reason", None) or finish
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            if getattr(delta, "content", None):
                parts.append(delta.content)
                visible = _visible("".join(parts))
                if len(visible) > len(shown):
                    on_text(visible[len(shown):])
                    shown = visible
            for tc in getattr(delta, "tool_calls", None) or []:
                key = tc.index if getattr(tc, "index", None) is not None else (tc.id or len(order))
                call = calls.get(key)
                if call is None:
                    call = calls[key] = ToolCall(id=getattr(tc, "id", "") or "", name="", arguments="")
                    order.append(key)
                if getattr(tc, "id", None) and not call.id:
                    call.id = tc.id
                function = getattr(tc, "function", None)
                if function is not None:
                    if getattr(function, "name", None) and not call.name:
                        call.name = function.name
                    call.arguments += getattr(function, "arguments", None) or ""
    tool_calls = [ToolCall(id=c.id or f"call_{i}", name=c.name, arguments=c.arguments or "{}")
                  for i, c in enumerate(calls[k] for k in order) if c.name]
    return Completion(text=_THINK.sub("", "".join(parts)).strip(), tool_calls=tool_calls, usage=usage, finish_reason=finish)


class LitellmProvider:
    def __init__(
        self,
        chain: list[dict[str, Any]],
        *,
        max_tokens: int = LIMITS.completion_max_tokens,
        temperature: float = 0.0,
        timeout_s: int = LIMITS.completion_timeout_s,
        effort: str = "",
        stream: bool = True,
    ) -> None:
        if not chain:
            raise ValueError("LitellmProvider needs at least one provider in the chain")
        self._chain = chain
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout_s = timeout_s
        self.effort = effort
        self.streams = stream

    @property
    def model(self) -> str:
        return self._chain[0]["litellm_model"]

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def complete(self, messages: list[Message], tools: list[dict[str, Any]], on_text: Callable[[str], None] | None = None) -> Completion:
        last_err: Exception | None = None
        streaming = self.streams and on_text is not None
        for provider in self._chain:
            kwargs: dict[str, Any] = {
                "model": provider["litellm_model"],
                "messages": to_wire(messages),
                "api_key": provider.get("api_key"),
                "api_base": provider.get("api_base"),
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "timeout": self._timeout_s,
                "stream": streaming,
            }
            if streaming:
                kwargs["stream_options"] = {"include_usage": True}
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
                response = _completion(**kwargs)
                return collect(response, on_text) if streaming else parse_response(response)
            except Exception as e:
                last_err = e
        raise RuntimeError(f"all providers failed: {last_err}")
