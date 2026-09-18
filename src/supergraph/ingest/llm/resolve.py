from __future__ import annotations

import os
import secrets

OLLAMA_CLOUD_BASE = "https://ollama.com/v1"
LOCAL_BASE_ENV = "LOCAL_LLM_BASE_URL"
LOCAL_KEY_ENV = "LOCAL_LLM_API_KEY"
# Any OpenAI-compatible server (llama-server, vLLM, LM Studio, Ollama) ignores or does not
# require a key; the chain drops keyless entries, so a placeholder stands in when none is set.
LOCAL_NO_KEY = "local"
OPENCODE_ZEN_BASE = "https://opencode.ai/zen/v1"
OPENCODE_GO_MARKER = "zen/go"
OPENCODE_SESSION_HEADER = "x-opencode-session"
_OPENCODE_SESSION = ""


def _opencode_session() -> str:
    global _OPENCODE_SESSION
    override = os.getenv("OPENCODE_SESSION", "").strip()
    if override:
        return override
    if not _OPENCODE_SESSION:
        _OPENCODE_SESSION = "ses_" + secrets.token_hex(16)
    return _OPENCODE_SESSION

DEFAULT_ALIASES: dict[str, str] = {
    "gpt-4": "groq/llama-3.3-70b-versatile",
    "gpt-4o": "groq/llama-3.3-70b-versatile",
    "gpt-4o-mini": "groq/llama-3.1-8b-instant",
    "gpt-3.5-turbo": "groq/llama-3.1-8b-instant",
    "claude-3-5-sonnet": "groq/llama-3.3-70b-versatile",
    "claude-3-5-haiku": "groq/llama-3.1-8b-instant",
}

DEFAULT_FREE_FIRST_CHAIN: list[str] = [
    "groq/llama-3.3-70b-versatile",
    "cerebras/llama-3.3-70b",
    "cloudflare/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "aistudio/gemini-2.0-flash",
    "openrouter/meta-llama/llama-3.3-70b-instruct",
]

_FREE_PREFIXES = ("groq/", "cerebras/", "cloudflare/", "aistudio/")


def resolve_model(model_id: str, aliases: dict[str, str] | None = None) -> dict:
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
    if model_id.startswith("opencode/"):
        slug = model_id[len("opencode/"):]
        base = os.getenv("OPENCODE_API_BASE", "").strip() or OPENCODE_ZEN_BASE
        entry = {"litellm_model": f"openai/{slug}", "api_base": base, "api_key": os.getenv("OPENCODE_API_KEY", "")}
        if OPENCODE_GO_MARKER in base:
            entry["extra_headers"] = {OPENCODE_SESSION_HEADER: _opencode_session()}
        return entry
    if model_id.startswith("local/"):
        slug = model_id[len("local/"):]
        base = os.getenv(LOCAL_BASE_ENV, "").strip().rstrip("/")
        return {"litellm_model": f"openai/{slug}", "api_base": base or None,
                "api_key": (os.getenv(LOCAL_KEY_ENV, "").strip() or LOCAL_NO_KEY) if base else ""}
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
        if r.get("extra_headers"):
            entry["extra_headers"] = r["extra_headers"]
        chain.append(entry)
    return chain
