from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from superclaw.catalog import learned
from superclaw.runtime import Usage
from superclaw.settings import PRICED_FILE, LIMITS


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


def _priced_path(cache_dir: Path) -> Path:
    return cache_dir / PRICED_FILE


def priced(model: str, cache_dir: Path | None) -> ModelInfo | None:
    if cache_dir is None:
        return None
    try:
        entry = json.loads(_priced_path(cache_dir).read_text()).get(model)
    except (OSError, ValueError):
        return None
    if not isinstance(entry, dict) or time.time() - float(entry.get("at") or 0) > LIMITS.models_cache_ttl_s:
        return None
    return ModelInfo(id=model, context_window=int(entry["context_window"]), max_output_tokens=int(entry["max_output_tokens"]),
                     input_per_token=float(entry["input_per_token"]), output_per_token=float(entry["output_per_token"]),
                     cache_read_per_token=float(entry["cache_read_per_token"]), known=bool(entry["known"]))


def remember_price(info: ModelInfo, cache_dir: Path | None) -> ModelInfo:
    if cache_dir is None:
        return info
    path = _priced_path(cache_dir)
    try:
        known = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        known = {}
    known[info.id] = {**{k: v for k, v in asdict(info).items() if k != "id"}, "at": time.time()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(known))
    return info


def lookup(model: str, cache_dir: Path | None = None) -> ModelInfo:
    if (cached := priced(model, cache_dir)) is not None:
        return cached
    return remember_price(_resolve(model, cache_dir), cache_dir)


def _resolve(model: str, cache_dir: Path | None = None) -> ModelInfo:
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
    seen = learned(model, cache_dir) if cache_dir else None
    if seen and seen.context_window:
        return ModelInfo(id=model, context_window=seen.context_window, max_output_tokens=LIMITS.max_output_tokens_fallback,
                         input_per_token=seen.input_per_token, output_per_token=seen.output_per_token, known=True)
    return ModelInfo(id=model, context_window=LIMITS.context_window_fallback, max_output_tokens=LIMITS.max_output_tokens_fallback)
