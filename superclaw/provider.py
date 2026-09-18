from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from typing import Any

from superclaw.reasoning import request_extras
from superclaw.runtime import Cancelled, Completion, Message, ToolCall, Usage, to_wire
from superclaw.settings import ERROR_HINTS, LIMITS

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
# A server that still rejects the resolved effort (detection missed, or an odd template) fails the
# whole request. The effort is a preference, the turn is not, so the call is retried once with the
# reasoning kwargs stripped.
_EFFORT_REJECTED = re.compile(r"reasoning[_ ]effort", re.IGNORECASE)
_REASONING_KEYS = ("reasoning_effort", "chat_template_kwargs")


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


def with_deadline(call: Callable[[], Any], seconds: float) -> Any:
    outcome: dict[str, Any] = {}

    def attempt() -> None:
        try:
            outcome["value"] = call()
        except BaseException as e:
            outcome["error"] = e

    worker = threading.Thread(target=attempt, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError(f"the provider timed out: it accepted the request and sent nothing for {seconds:g}s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def parse_response(resp: Any) -> Completion:
    choice = resp.choices[0]
    message = choice.message
    reasoning = getattr(message, "reasoning_content", None) or ""
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
        reasoning=reasoning,
    )


def _visible(text: str) -> str:
    cleaned = _THINK.sub("", text)
    start = cleaned.find("<think>")
    return cleaned if start < 0 else cleaned[:start]


def collect(chunks: Any, on_text: Callable[[str], None], cancelled: Callable[[], bool] | None = None, stall_s: float = 0.0,
            on_reasoning: Callable[[str], None] | None = None) -> Completion:
    parts: list[str] = []
    thoughts: list[str] = []
    calls: dict[Any, ToolCall] = {}
    order: list[Any] = []
    shown, finish, usage = "", "", Usage()
    deadline = time.monotonic() + stall_s if stall_s else 0.0
    for chunk in chunks:
        if cancelled is not None and cancelled():
            raise Cancelled("the user stopped the run")
        if deadline and time.monotonic() > deadline:
            raise TimeoutError(f"the provider timed out: the stream stalled for {stall_s:g}s")
        deadline = time.monotonic() + stall_s if stall_s else 0.0
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
            if getattr(delta, "reasoning_content", None):
                thoughts.append(delta.reasoning_content)
                if on_reasoning is not None:
                    on_reasoning(delta.reasoning_content)
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
    return Completion(text=_THINK.sub("", "".join(parts)).strip(), tool_calls=tool_calls, usage=usage,
                      finish_reason=finish, reasoning="".join(thoughts))


class LitellmProvider:
    def __init__(
        self,
        chain: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]],
        *,
        max_tokens: int = LIMITS.completion_max_tokens,
        temperature: float = 0.0,
        timeout_s: int = LIMITS.completion_timeout_s,
        effort: str = "",
        stream: bool = True,
    ) -> None:
        self._source = chain
        self._resolved: list[dict[str, Any]] | None = chain if isinstance(chain, list) else None
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout_s = timeout_s
        self.effort = effort
        self.streams = stream

    @property
    def _chain(self) -> list[dict[str, Any]]:
        if self._resolved is None:
            self._resolved = self._source() if callable(self._source) else self._source
        if not self._resolved:
            raise RuntimeError("no API key resolved for this model; run `superclaw setup` or set the provider's key")
        return self._resolved

    @property
    def model(self) -> str:
        return self._chain[0]["litellm_model"]

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def complete(self, messages: list[Message], tools: list[dict[str, Any]], on_text: Callable[[str], None] | None = None,
                 max_tokens: int | None = None, cancelled: Callable[[], bool] | None = None,
                 on_reasoning: Callable[[str], None] | None = None) -> Completion:
        last_err: Exception | None = None
        streaming = self.streams and on_text is not None
        for provider in self._chain:
            kwargs: dict[str, Any] = {
                "model": provider["litellm_model"],
                "messages": to_wire(messages),
                "api_key": provider.get("api_key"),
                "api_base": provider.get("api_base"),
                "temperature": self._temperature,
                "max_tokens": max_tokens or self._max_tokens,
                "timeout": self._timeout_s,
                "stream": streaming,
            }
            if streaming:
                kwargs["stream_options"] = {"include_usage": True}
            if provider.get("account_id"):
                kwargs["account_id"] = provider["account_id"]
            kwargs.update(request_extras(self.effort, provider.get("api_base"), None))
            if provider.get("extra_headers"):
                kwargs["extra_headers"] = provider["extra_headers"]
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            try:
                return self._call(kwargs, on_text, cancelled, streaming, on_reasoning)
            except Cancelled:
                raise
            except Exception as e:
                last_err = e
        raise RuntimeError(f"all providers failed: {last_err}")

    def _call(self, kwargs: dict[str, Any], on_text: Callable[[str], None] | None, cancelled: Callable[[], bool] | None,
              streaming: bool, on_reasoning: Callable[[str], None] | None = None) -> Completion:
        try:
            response = with_deadline(lambda: _completion(**kwargs), self._timeout_s)
        except Cancelled:
            raise
        except Exception as e:
            if not any(k in kwargs for k in _REASONING_KEYS) or not _EFFORT_REJECTED.search(str(e)):
                raise
            response = with_deadline(lambda: _completion(**{k: v for k, v in kwargs.items() if k not in _REASONING_KEYS}), self._timeout_s)
        if streaming:
            return collect(response, on_text, cancelled, self._timeout_s, on_reasoning)
        return parse_response(response)
