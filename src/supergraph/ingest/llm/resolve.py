"""Provider-prefix model resolution for cloud NL ingestion.

Ports zerocostllm's resolve_model + free-first ordering, kept self-contained
(reads os.environ directly, no tools/ or repo-root env.py dependency) so it
works from an installed wheel. Turns a provider-prefixed model id into the
call kwargs LLMRunner needs, and builds an ordered free-first provider chain.
"""
from __future__ import annotations

import os

OLLAMA_CLOUD_BASE = "https://ollama.com/v1"

# OpenAI/Anthropic names route to free equivalents. Override via IngestConfig.
DEFAULT_ALIASES: dict[str, str] = {
    "gpt-4": "groq/llama-3.3-70b-versatile",
    "gpt-4o": "groq/llama-3.3-70b-versatile",
    "gpt-4o-mini": "groq/llama-3.1-8b-instant",
    "gpt-3.5-turbo": "groq/llama-3.1-8b-instant",
    "claude-3-5-sonnet": "groq/llama-3.3-70b-versatile",
    "claude-3-5-haiku": "groq/llama-3.1-8b-instant",
}

# Free-tier-first default candidate chain (provider-prefixed ids).
DEFAULT_FREE_FIRST_CHAIN: list[str] = [
    "groq/llama-3.3-70b-versatile",
    "cerebras/llama-3.3-70b",
    "cloudflare/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "aistudio/gemini-2.0-flash",
    "openrouter/meta-llama/llama-3.3-70b-instruct",
]

_FREE_PREFIXES = ("groq/", "cerebras/", "cloudflare/", "aistudio/")


def resolve_model(model_id: str, aliases: dict[str, str] | None = None) -> dict:
    """Resolve a provider-prefixed model id to litellm call kwargs.

    Returns {litellm_model, api_base, api_key, [account_id]}. api_key may be
    "" when the provider env var is unset; build_provider_chain drops those.
    """
    aliases = aliases if aliases is not None else DEFAULT_ALIASES
    model_id = aliases.get(model_id, model_id)

    if model_id.startswith("groq/"):
        slug = model_id[len("groq/"):]
        return {"litellm_model": f"groq/{slug}", "api_base": None,
                "api_key": os.getenv("GROQ_API_KEY", "")}
    if model_id.startswith("cerebras/"):
        slug = model_id[len("cerebras/"):]
        return {"litellm_model": f"cerebras/{slug}", "api_base": None,
                "api_key": os.getenv("CEREBRAS_API_KEY", "")}
    if model_id.startswith("cloudflare/"):
        slug = model_id[len("cloudflare/"):]
        return {"litellm_model": f"cloudflare/{slug}", "api_base": None,
                "api_key": os.getenv("CLOUDFLARE_API_KEY", ""),
                "account_id": os.getenv("CLOUDFLARE_ACCOUNT_ID", "")}
    if model_id.startswith("aistudio/"):
        slug = model_id[len("aistudio/"):]
        return {"litellm_model": f"gemini/{slug}", "api_base": None,
                "api_key": os.getenv("GOOGLE_AISTUDIO_API_KEY", "")}
    if model_id.startswith("nvidia_nim/"):
        slug = model_id[len("nvidia_nim/"):]
        return {"litellm_model": f"nvidia_nim/{slug}", "api_base": None,
                "api_key": os.getenv("NVIDIA_NIM_API_KEY", "")}
    if model_id.startswith("ollama/"):
        slug = model_id[len("ollama/"):]
        return {"litellm_model": f"openai/{slug}", "api_base": OLLAMA_CLOUD_BASE,
                "api_key": os.getenv("OLLAMA_API_KEY", "ollama")}
    if model_id.startswith("openrouter/"):
        return {"litellm_model": model_id, "api_base": None,
                "api_key": os.getenv("OPENROUTER_API_KEY", "")}
    return {"litellm_model": f"openrouter/{model_id}", "api_base": None,
            "api_key": os.getenv("OPENROUTER_API_KEY", "")}


def build_provider_chain(
    models: list[str],
    *,
    free_first: bool = True,
    aliases: dict[str, str] | None = None,
) -> list[dict]:
    """Map model ids to LLMRunner provider dicts.

    Drops entries whose provider has no API key set. With free_first, stable-
    sorts free-prefixed providers ahead of paid ones (preserving input order
    within each group). Cloudflare entries also carry account_id.
    """
    ordered = list(models)
    if free_first:
        ordered = sorted(
            ordered,
            key=lambda m: 0 if m.startswith(_FREE_PREFIXES) else 1,
        )
    chain: list[dict] = []
    for m in ordered:
        r = resolve_model(m, aliases)
        if not r["api_key"]:
            continue
        entry = {
            "pid": m,
            "litellm_model": r["litellm_model"],
            "api_base": r.get("api_base"),
            "api_key": r["api_key"],
        }
        if r.get("account_id"):
            entry["account_id"] = r["account_id"]
        chain.append(entry)
    return chain
