"""Model-dependent reasoning-effort resolution.

superclaw exposes one abstract effort (off / low / medium / high). Each model's server accepts
its own vocabulary: OpenAI takes low|medium|high, Qwen3.x/Bonsai take low|medium|xhigh and turn
thinking off through ``enable_thinking``, others differ again. Sending the abstract level verbatim
gets rejected (or, for ``off``, silently omitted so the model falls back to its most expensive
default). This module reads what a model actually accepts from its own chat template and maps the
abstract effort onto it, so ``high`` reaches the model as ``xhigh`` and ``off`` truly disables
thinking. Servers with no introspection endpoint fall through to pass-the-level-through, and a
call-time rejection is still caught in ``LitellmProvider`` as the last resort.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from urllib.request import urlopen

# Ascending intensity. The abstract effort names superclaw uses (low/medium/high) live here too, so
# a level and an accepted level are comparable by index.
LADDER = ("minimal", "low", "medium", "high", "xhigh", "max")
_OFF = ("", "off")


def _rank(level: str) -> int:
    return LADDER.index(level) if level in LADDER else -1


def resolve_reasoning(effort: str, accepted: tuple[str, ...], can_disable: bool) -> dict:
    """Abstract effort -> the request kwargs a model with this ``accepted`` vocabulary understands.

    ``accepted`` empty means the model was not introspectable: pass the level through unchanged and
    let the server (and the reject-retry) sort it out. ``off`` disables thinking when the template
    supports it, else drops to the lowest accepted level, else omits the parameter.
    """
    accepted = tuple(a for a in accepted if a in LADDER)
    if effort in _OFF:
        if can_disable:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        if accepted:
            return {"reasoning_effort": min(accepted, key=_rank)}
        return {}
    if _rank(effort) < 0:
        return {"reasoning_effort": effort} if effort else {}
    if not accepted or effort in accepted:
        return {"reasoning_effort": effort}
    at_or_above = [a for a in accepted if _rank(a) >= _rank(effort)]
    chosen = min(at_or_above, key=_rank) if at_or_above else max(accepted, key=_rank)
    return {"reasoning_effort": chosen}


def _props_url(api_base: str) -> str:
    root = api_base.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root.rstrip("/") + "/props"


def _DEFAULT_FETCH(url: str) -> bytes:
    with urlopen(url, timeout=4) as response:
        return response.read()


def detect_reasoning(api_base: str, fetch: Callable[[str], bytes] | None = None) -> tuple[tuple[str, ...], bool]:
    """(accepted levels, thinking-can-be-disabled) read from the server's chat template, or ((), False)
    when the endpoint is missing or the model is not a reasoning model."""
    if not api_base:
        return (), False
    try:
        template = json.loads((fetch or _DEFAULT_FETCH)(_props_url(api_base))).get("chat_template", "")
    except Exception:
        return (), False
    if "reasoning_effort" not in template:
        return (), False
    match = re.search(r"reasoning_effort[^\n]*?not in \(([^)]*)\)", template) or re.search(r"not in \(([^)]*)\)", template)
    accepted = tuple(x.strip().strip("'\"") for x in match.group(1).split(",")) if match else ()
    accepted = tuple(a for a in accepted if a in LADDER)
    return accepted, "enable_thinking" in template


_CACHE: dict[str, tuple[tuple[str, ...], bool]] = {}


def request_extras(effort: str, api_base: str | None, fetch: Callable[[str], bytes] | None,
                   cache: dict[str, tuple[tuple[str, ...], bool]] | None = None) -> dict:
    """The reasoning kwargs for one call. Detection is cached per base URL. With no base URL (a cloud
    provider) the abstract level is passed straight through."""
    if not api_base:
        return {"reasoning_effort": effort} if effort and effort not in _OFF else {}
    store = _CACHE if cache is None else cache
    if api_base not in store:
        store[api_base] = detect_reasoning(api_base, fetch)
    accepted, can_disable = store[api_base]
    return resolve_reasoning(effort, accepted, can_disable)
