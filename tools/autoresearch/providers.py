from __future__ import annotations

import json
from pathlib import Path

from env import ENV

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def resolve_providers(
    config: dict,
    model_priority: list[str] | None = None,
) -> list[dict]:
    providers = config.get("providers", {})
    active_pid = config.get("active_provider", "")
    provider_order = [active_pid] + [
        p for p in config.get("provider_fallback_order", []) if p != active_pid
    ]
    provider_order = [p for p in dict.fromkeys(provider_order) if p in providers]

    result: list[dict] = []
    for pid in provider_order:
        p = providers.get(pid)
        if not p:
            continue
        base_url = p.get("base_url", "")
        if not base_url:
            continue
        is_local = p.get("is_local", "localhost" in base_url or "127.0.0.1" in base_url)
        env_field = p.get("env_key", "")
        if not env_field:
            if not is_local:
                raise ValueError(
                    f"provider {pid!r} in config.json is not local and is missing "
                    f"'env_key'. Add 'env_key' pointing at an env.py field, or mark "
                    f"the provider is_local=true."
                )
            api_key = "ollama"
        else:
            api_key = str(ENV[env_field])
        prefix = p.get("litellm_prefix") or ("ollama_chat" if is_local else "")
        available = p.get("models", {})

        if model_priority is not None:
            selected = [m for m in model_priority if m in available][:1]
        else:
            active_model = config.get("active_model", "")
            fallbacks = list(p.get("model_fallback_order", []))
            order = [active_model] + fallbacks
            selected = [m for m in dict.fromkeys(order) if m and m in available]

        for model in selected:
            litellm_model = f"{prefix}/{model}" if prefix else model
            result.append({
                "pid": pid,
                "litellm_model": litellm_model,
                "api_base": base_url,
                "api_key": api_key,
            })

    return result
