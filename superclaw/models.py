from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from superclaw.runtime import Usage
from superclaw.settings import LIMITS

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


@dataclass(frozen=True)
class ModelInfo:
    id: str
    context_window: int
    max_output_tokens: int
    input_per_token: float = 0.0
    output_per_token: float = 0.0
    cache_read_per_token: float = 0.0
    known: bool = False

    def cost(self, usage: Usage) -> float:
        cached = min(usage.cache_read_tokens, usage.input_tokens)
        fresh = usage.input_tokens - cached
        return fresh * self.input_per_token + cached * self.cache_read_per_token + usage.output_tokens * self.output_per_token


def _candidates(model: str) -> Iterator[str]:
    yield model
    parts = model.split("/")
    for i in range(1, len(parts)):
        yield "/".join(parts[i:])


def _catalog() -> dict[str, dict[str, Any]]:
    import litellm

    return litellm.model_cost


def lookup(model: str) -> ModelInfo:
    catalog = _catalog()
    for candidate in _candidates(model):
        entry = catalog.get(candidate)
        if entry and entry.get("max_input_tokens"):
            return ModelInfo(
                id=model,
                context_window=int(entry["max_input_tokens"]),
                max_output_tokens=int(entry.get("max_output_tokens") or LIMITS.max_output_tokens_fallback),
                input_per_token=float(entry.get("input_cost_per_token") or 0.0),
                output_per_token=float(entry.get("output_cost_per_token") or 0.0),
                cache_read_per_token=float(entry.get("cache_read_input_token_cost") or 0.0),
                known=True,
            )
    return ModelInfo(id=model, context_window=LIMITS.context_window_fallback, max_output_tokens=LIMITS.max_output_tokens_fallback)
