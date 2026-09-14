from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from superclaw.settings import CATALOG_CHAT_MODE, LIMITS, MODEL_SOURCE_CATALOG, MODEL_SOURCE_LIVE, NONCODING_TERMS, PRICE_UNIT_TOKENS, PROVIDERS, Provider
from superclaw.text import compact

Fetch = Callable[[str, dict[str, str]], bytes]


@dataclass(frozen=True)
class Model:
    id: str
    provider: str
    name: str
    context_window: int
    input_per_token: float
    output_per_token: float
    tools: bool
    source: str


def provider_of(model: str) -> Provider | None:
    return next((p for p in PROVIDERS if model.startswith(p.name + "/")), None)


def keyed_providers() -> list[Provider]:
    return [p for p in PROVIDERS if os.environ.get(p.env)]


def coding(slug: str) -> bool:
    low = slug.lower()
    return not any(term in low for term in NONCODING_TERMS)


def catalog_models(provider: Provider) -> list[Model]:
    if not provider.prefix:
        return []
    import litellm

    head = provider.prefix + "/"
    out = []
    for key, entry in litellm.model_cost.items():
        slug = key[len(head):]
        if not key.startswith(head) or not entry.get("max_input_tokens") or entry.get("mode") != CATALOG_CHAT_MODE or not coding(slug):
            continue
        out.append(Model(id=f"{provider.name}/{slug}", provider=provider.name, name=slug, context_window=int(entry["max_input_tokens"]),
                         input_per_token=float(entry.get("input_cost_per_token") or 0.0), output_per_token=float(entry.get("output_cost_per_token") or 0.0),
                         tools=bool(entry.get("supports_function_calling")), source=MODEL_SOURCE_CATALOG))
    return sorted(out, key=lambda m: m.id)


def parse_models(provider: Provider, body: bytes) -> list[Model]:
    doc: dict[str, Any] = json.loads(body)
    out = []
    for item in doc.get("data") or doc.get("models") or []:
        slug = str(item.get("id") or item.get("name") or "").removeprefix("models/")
        methods = item.get("supportedGenerationMethods") or []
        if not slug or not coding(slug) or (methods and "generateContent" not in methods):
            continue
        label = str(item.get("displayName") or item.get("name") or slug)
        pricing = item.get("pricing") or {}
        out.append(Model(id=f"{provider.name}/{slug}", provider=provider.name, name=slug if label.startswith("models/") else label,
                         context_window=int(item.get("context_length") or item.get("context_window") or item.get("inputTokenLimit") or 0),
                         input_per_token=float(pricing.get("prompt") or 0.0), output_per_token=float(pricing.get("completion") or 0.0),
                         tools="tools" in (item.get("supported_parameters") or []), source=MODEL_SOURCE_LIVE))
    return sorted(out, key=lambda m: m.id)


def _get(url: str, headers: dict[str, str]) -> bytes:
    response = httpx.get(url, headers=headers, timeout=LIMITS.models_fetch_timeout_s)
    response.raise_for_status()
    return response.content[: LIMITS.models_fetch_bytes]


def live_models(provider: Provider, key: str, fetch: Fetch | None = None) -> list[Model]:
    if not provider.models_url or not (key or provider.public):
        return []
    url = f"{provider.models_url}?key={key}" if provider.key_in_query else provider.models_url
    headers = {"Authorization": f"Bearer {key}"} if key and not provider.key_in_query else {}
    return parse_models(provider, (fetch or _get)(url, headers))


def cached_models(provider: Provider, cache_dir: Path) -> list[Model]:
    path = cache_dir / f"{provider.name}.json"
    try:
        if time.time() - path.stat().st_mtime > LIMITS.models_cache_ttl_s:
            return []
        return [Model(**row) for row in json.loads(path.read_text())]
    except (OSError, ValueError, TypeError):
        return []


def models_for(provider: Provider, key: str, cache_dir: Path, refresh: bool = False, online: bool = True, fetch: Fetch | None = None) -> list[Model]:
    live = [] if refresh else cached_models(provider, cache_dir)
    if not live and online:
        try:
            live = live_models(provider, key, fetch)
        except (httpx.HTTPError, ValueError, OSError):
            live = []
        if live:
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / f"{provider.name}.json").write_text(json.dumps([asdict(m) for m in live]))
    return merge(live, catalog_models(provider))


def merge(live: list[Model], catalog: list[Model]) -> list[Model]:
    if not live:
        return catalog
    known = {m.id: m for m in catalog}
    out = []
    for model in live:
        fill = known.get(model.id)
        out.append(model if fill is None else Model(
            model.id, model.provider, model.name, model.context_window or fill.context_window,
            model.input_per_token or fill.input_per_token, model.output_per_token or fill.output_per_token,
            model.tools or fill.tools, model.source))
    return out


def learned(model_id: str, cache_dir: Path) -> Model | None:
    provider = provider_of(model_id)
    if provider is None:
        return None
    return next((m for m in cached_models(provider, cache_dir) if m.id == model_id), None)


def describe(model: Model, dot: str) -> str:
    parts = [f"{compact(model.context_window)} ctx"] if model.context_window else []
    if model.tools:
        parts.append("tools")
    if model.input_per_token or model.output_per_token:
        parts.append(f"${model.input_per_token * PRICE_UNIT_TOKENS:.2f}/{model.output_per_token * PRICE_UNIT_TOKENS:.2f}")
    parts.append(model.source)
    return f" {dot} ".join(parts)


def resolve(text: str, models: list[Model], active_provider: str) -> Model | None:
    low = text.strip().lower()
    by_id = {m.id.lower(): m for m in models}
    for candidate in (low, f"{active_provider}/{low}"):
        if candidate in by_id:
            return by_id[candidate]
    for hits in ([m for m in models if m.id.lower().split("/", 1)[1] == low], [m for m in models if low in m.id.lower()]):
        if len(hits) == 1:
            return hits[0]
    return None
