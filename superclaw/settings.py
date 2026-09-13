from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-flash"
DEFAULT_MODE = "ask"
EFFORTS = ("low", "medium", "high")
EFFORT_OFF = "off"
CREDENTIALS_FILE = "credentials.env"
CREDENTIALS_MODE = 0o600
TRANSCRIPT_TEMPLATE = "superclaw-transcript-{sid}.md"
ASCII_ENV = "SUPERCLAW_ASCII"
OFF_VALUES = ("", "0", "false", "off")
LOCALE_ENVS = ("LC_ALL", "LC_CTYPE", "LANG")
UTF8_MARK = "utf"


@dataclass(frozen=True)
class Glyphs:
    prompt: str
    running: str
    ok: str
    failed: str
    dot: str
    gauge: str
    mode: str
    child: str
    ellipsis: str
    call: str
    spinner: str
    border: str
    block_art: bool


UNICODE = Glyphs(prompt="\u276f", running="\u25d0", ok="\u2713", failed="\u2717", dot="\u00b7", gauge="\u25d4", mode="\u25cf", child="\u21b3", ellipsis="\u2026", call="\u2192",
                 spinner="\u25d0\u25d3\u25d1\u25d2", border="round", block_art=True)
ASCII = Glyphs(prompt=">", running="~", ok="+", failed="x", dot="|", gauge="#", mode="*", child="->", ellipsis="...", call="->",
               spinner="-\\|/", border="ascii", block_art=False)


def choose_glyphs(e: Mapping[str, str]) -> Glyphs:
    if e.get(ASCII_ENV, "").strip().lower() not in OFF_VALUES:
        return ASCII
    locale = next((e[name] for name in LOCALE_ENVS if e.get(name)), "")
    return UNICODE if UTF8_MARK in locale.lower() else ASCII


@dataclass(frozen=True)
class Provider:
    name: str
    env: str
    default_model: str
    console: str
    prefix: str
    models_url: str
    public: bool = False
    key_in_query: bool = False


PROVIDERS = (
    Provider("openrouter", "OPENROUTER_API_KEY", DEFAULT_MODEL, "https://openrouter.ai/keys", "openrouter", "https://openrouter.ai/api/v1/models", public=True),
    Provider("groq", "GROQ_API_KEY", "groq/llama-3.3-70b-versatile", "https://console.groq.com/keys", "groq", "https://api.groq.com/openai/v1/models"),
    Provider("cerebras", "CEREBRAS_API_KEY", "cerebras/llama-3.3-70b", "https://cloud.cerebras.ai", "cerebras", "https://api.cerebras.ai/v1/models"),
    Provider("ollama", "OLLAMA_API_KEY", "ollama/gpt-oss:120b", "https://ollama.com/settings/keys", "", "https://ollama.com/v1/models"),
    Provider("aistudio", "GOOGLE_AISTUDIO_API_KEY", "aistudio/gemini-2.0-flash", "https://aistudio.google.com/apikey", "gemini",
             "https://generativelanguage.googleapis.com/v1beta/models", key_in_query=True),
    Provider("nvidia_nim", "NVIDIA_NIM_API_KEY", "nvidia_nim/meta/llama-3.3-70b-instruct", "https://build.nvidia.com", "nvidia_nim",
             "https://integrate.api.nvidia.com/v1/models", public=True),
    Provider("opencode", "OPENCODE_API_KEY", "opencode/deepseek-v4-flash", "https://opencode.ai/auth", "opencode", "https://opencode.ai/zen/v1/models"),
)
CATALOG_CHAT_MODE = "chat"
NONCODING_TERMS = ("audio", "dall-e", "deep-research", "embed", "image", "imagen", "moderation", "realtime", "rerank", "sora", "speech",
                   "transcribe", "translate", "tts", "veo", "whisper", "aqa", "guard", "safeguard")
MODEL_SOURCE_LIVE = "live"
MODEL_SOURCE_CATALOG = "catalog"
MODELS_CACHE_DIR = "models"


@dataclass(frozen=True)
class ErrorHint:
    needles: tuple[str, ...]
    tui: str
    cli: str


ERROR_HINTS = (
    ErrorHint(("invalid api key", "invalid_api_key", "unauthorized", "authentication", "api key", "401", "403"),
              "key rejected; /setup stores a new one", "key rejected; run `superclaw setup`"),
    ErrorHint(("rate limit", "rate_limit", "too many requests", "quota", "overloaded", "resource_exhausted", "429", "529"),
              "rate limited; wait, or /model picks another", "rate limited; wait, or pass --model"),
    ErrorHint(("context length", "context window", "maximum context", "too many tokens", "prompt is too long", "reduce the length"),
              "context window full; /new starts fresh", "context window full; shorten the prompt or drop --resume"),
    ErrorHint(("not a valid model", "model not found", "model_not_found", "does not exist", "unknown model", "no such model", "unsupported model",
               "invalid model", "model is not"),
              "model unavailable; /model lists what the provider serves", "model unavailable; `superclaw models` lists what the provider serves"),
    ErrorHint(("connection", "timed out", "timeout", "no such host", "name resolution", "unreachable", "tls", "reset by peer", "dns"),
              "provider unreachable; check the network", "provider unreachable; check the network"),
)


OPENCODE_KEY_ENV = "OPENCODE_API_KEY"
OPENCODE_BASE_ENV = "OPENCODE_API_BASE"
OPENCODE_GO_BASE = "https://opencode.ai/zen/go/v1"
OPENCODE_AUTH_FILE_KEY = "opencode-go"
OPENCODE_AUTH_TYPE = "api"


def opencode_auth_path(e: Mapping[str, str]) -> Path:
    if override := e.get("OPENCODE_AUTH_PATH", "").strip():
        return Path(override)
    if directory := e.get("OPENCODE_DIR", "").strip():
        return Path(directory) / "auth.json"
    if xdg := e.get("XDG_DATA_HOME", "").strip():
        return Path(xdg) / "opencode" / "auth.json"
    return Path.home() / ".local" / "share" / "opencode" / "auth.json"


def read_opencode_key(e: Mapping[str, str]) -> str:
    try:
        data = json.loads(opencode_auth_path(e).read_text())
    except (OSError, ValueError):
        return ""
    entry = data.get(OPENCODE_AUTH_FILE_KEY) if isinstance(data, dict) else None
    if isinstance(entry, dict) and str(entry.get("type", "")).strip().lower() == OPENCODE_AUTH_TYPE:
        return str(entry.get("key", "")).strip()
    return ""


def read_env_file(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    pairs = (line.partition("=") for line in lines if "=" in line and not line.lstrip().startswith("#"))
    return {key.strip(): value.strip().strip("'\"") for key, _, value in pairs if key.strip()}


@dataclass(frozen=True)
class Limits:
    context_window_fallback: int = 128_000
    max_output_tokens_fallback: int = 4096
    message_overhead_tokens: int = 4
    completion_max_tokens: int = 4096
    completion_timeout_s: int = 120
    max_turns: int = 12
    usd_decimals: int = 6

    compaction_reserve_tokens: int = 16_384
    compaction_keep_tokens: int = 20_000
    compaction_window_share: float = 0.25
    prune_threshold_chars: int = 8192
    prune_head_chars: int = 4096
    prune_tail_chars: int = 1024
    compaction_tool_result_clamp: int = 2000
    compaction_tool_args_clamp: int = 500
    compaction_user_words: int = 256
    compaction_assistant_words: int = 200
    compaction_calls_per_turn: int = 8
    compaction_arg_bytes: int = 120
    compaction_error_bytes: int = 150
    compaction_result_bytes: int = 1200
    compaction_previous_summary_bytes: int = 16 * 1024
    compaction_brief_max_bytes: int = 24 * 1024
    verifier_transcript_bytes: int = 24_000

    max_empty_turns: int = 3
    failure_hint_at: int = 2
    failure_stop_at: int = 6
    stale_plan_tool_calls: int = 10
    tool_only_reminder_at: int = 6
    max_continue_nudges: int = 3
    identical_call_at: int = 3
    max_calls_per_turn: int = 42

    hook_timeout_s: int = 60
    hook_block_exit_code: int = 2
    memory_recall_limit: int = 5
    guideline_file_bytes: int = 8 * 1024
    guideline_total_bytes: int = 32 * 1024
    skills_index_bytes: int = 4096
    skill_description_chars: int = 200

    compaction_head_share: float = 0.4
    memory_recall_max: int = 20
    error_signature_chars: int = 160
    hook_output_chars: int = 4000
    hook_error_chars: int = 200

    eager_schema_tokens: int = 1300
    obs_min_chars: int = 2048
    ref_hex_chars: int = 8
    ref_hex_step: int = 4
    recall_chunk_tokens: int = 4000
    recall_search_limit: int = 5
    recall_preview_chars: int = 120

    tool_output_bytes: int = 64 * 1024
    tool_output_tokens: int = 10_000
    read_file_tokens: int = 25_000
    truncation_notice_tokens: int = 60
    chars_per_token: int = 4
    budget_head_share: float = 0.6
    budget_head_lines: int = 10
    budget_tail_lines: int = 16
    budget_failure_context_before: int = 2
    budget_failure_context_after: int = 3
    diff_preview_bytes: int = 48 * 1024
    binary_sniff_bytes: int = 8192
    read_file_bytes: int = 256 * 1024
    read_file_lines: int = 2000
    read_line_chars: int = 2000
    list_directory_entries: int = 500
    list_directory_depth: int = 2
    list_directory_max_depth: int = 5
    glob_limit: int = 100
    glob_max_limit: int = 1000
    grep_head_limit: int = 50
    tool_search_matches: int = 10
    shell_timeout_ms: int = 60_000
    shell_max_timeout_ms: int = 600_000
    shell_capture_bytes: int = 1024 * 1024
    capture_preview_lines: int = 5
    kernel_trace_depth: int = 3
    kernel_checkpoint_timeout_s: float = 10.0
    delegate_refs_returned: int = 5
    delegate_depth: int = 2
    delegate_max_turns: int = 24
    delegate_budget_tokens: int = 200_000
    delegate_min_budget_tokens: int = 20_000
    delegate_handoff_tokens: int = 8000
    delegate_answer_tokens: int = 1500

    id_hash_chars: int = 16
    run_id_bytes: int = 4
    session_id_bytes: int = 3
    event_seq_digits: int = 6
    session_list_limit: int = 1000
    session_events_limit: int = 100_000
    recent_sessions_shown: int = 20
    recent_models_shown: int = 5
    models_fetch_timeout_s: float = 15.0
    models_fetch_bytes: int = 8 * 1024 * 1024
    models_cache_ttl_s: int = 86_400
    model_id_width: int = 44
    model_list_shown: int = 40
    tool_name_width: int = 16
    preview_args_chars: int = 160
    preview_error_chars: int = 200
    dialog_args_chars: int = 1200
    card_body_lines: int = 12
    card_arg_chars: int = 80
    gauge_cells: int = 8
    spinner_interval_s: float = 0.09
    timer_interval_s: float = 0.25
    command_matches_shown: int = 6
    tui_tier_narrow: int = 58
    tui_tier_medium: int = 80
    tui_tier_full: int = 100
    welcome_max_gap: int = 8


LIMITS = Limits()


@dataclass(frozen=True)
class Settings:
    model: str
    mode: str
    effort: str
    context_window: int
    budget_tokens: int
    budget_usd: float
    data_dir: Path
    config_dir: Path
    db_path: Path
    skills_dir: Path | None
    glyphs: Glyphs
    cache_dir: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        home = Path.home()
        data_dir = Path(e.get("XDG_DATA_HOME", "").strip() or home / ".local" / "share") / "superclaw"
        config_dir = Path(e.get("XDG_CONFIG_HOME", "").strip() or home / ".config") / "superclaw"
        cache_dir = Path(e.get("XDG_CACHE_HOME", "").strip() or home / ".cache") / "superclaw"
        saved = read_env_file(config_dir / CREDENTIALS_FILE)
        if not saved.get(OPENCODE_KEY_ENV) and not e.get(OPENCODE_KEY_ENV) and (ambient := read_opencode_key(e)):
            saved[OPENCODE_KEY_ENV] = ambient
            if not e.get(OPENCODE_BASE_ENV):
                saved[OPENCODE_BASE_ENV] = OPENCODE_GO_BASE
        if env is None:
            for key, value in saved.items():
                os.environ.setdefault(key, value)
        e = {**saved, **e}
        db_override = e.get("SUPERCLAW_DB_PATH", "").strip()
        skills_override = e.get("SUPERCLAW_SKILLS_DIR", "").strip()
        return cls(
            model=e.get("SUPERCLAW_MODEL", "").strip() or DEFAULT_MODEL,
            mode=e.get("SUPERCLAW_MODE", "").strip() or DEFAULT_MODE,
            effort=e.get("SUPERCLAW_EFFORT", "").strip().lower() if e.get("SUPERCLAW_EFFORT", "").strip().lower() in EFFORTS else "",
            context_window=int(e.get("SUPERCLAW_CONTEXT_WINDOW", "").strip() or 0),
            budget_tokens=int(e.get("SUPERCLAW_BUDGET_TOKENS", "").strip() or 0),
            budget_usd=float(e.get("SUPERCLAW_BUDGET_USD", "").strip() or 0),
            data_dir=data_dir,
            config_dir=config_dir,
            db_path=Path(db_override) if db_override else data_dir / "brain",
            skills_dir=Path(skills_override) if skills_override else None,
            glyphs=choose_glyphs(e),
            cache_dir=cache_dir,
        )

    @property
    def user_guidelines(self) -> Path:
        return self.config_dir / "SUPERCLAW.md"

    @property
    def user_hooks(self) -> Path:
        return self.config_dir / "hooks.json"

    @property
    def credentials(self) -> Path:
        return self.config_dir / CREDENTIALS_FILE

    @property
    def models_cache(self) -> Path:
        return self.cache_dir / MODELS_CACHE_DIR

    def save_credentials(self, provider: Provider, key: str, model: str) -> None:
        self._save({provider.env: key, "SUPERCLAW_MODEL": model})

    def save_model(self, model: str) -> None:
        self._save({"SUPERCLAW_MODEL": model})

    def save_effort(self, effort: str) -> None:
        self._save({"SUPERCLAW_EFFORT": effort})

    def _save(self, values: dict[str, str]) -> None:
        merged = {**read_env_file(self.credentials), **values}
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.credentials.write_text("".join(f"{k}={v}\n" for k, v in merged.items()))
        self.credentials.chmod(CREDENTIALS_MODE)
        os.environ.update(values)

    def model_info(self):
        from superclaw.models import lookup

        return lookup(self.model, self.models_cache)

    def window(self) -> int:
        return self.context_window or self.model_info().context_window

    def skill_roots(self, workspace: Path | None = None) -> list[Path]:
        roots = [self.skills_dir] if self.skills_dir else []
        roots += [self.config_dir / "skills", Path.home() / ".agents" / "skills"]
        if workspace is not None:
            roots.append(Path(workspace) / ".superclaw" / "skills")
        return roots
