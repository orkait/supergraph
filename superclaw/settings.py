from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superclaw.models import ModelInfo

DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-flash"
DEFAULT_MODE = "ask"
EFFORTS = ("low", "medium", "high")
EFFORT_OFF = "off"
CREDENTIALS_FILE = "credentials.env"
CREDENTIALS_MODE = 0o600
TRANSCRIPT_TEMPLATE = "superclaw-transcript-{sid}.md"
ASCII_ENV = "SUPERCLAW_ASCII"
OFF_VALUES = ("", "0", "false", "off")
SEARCH_ENV = "SUPERCLAW_SEARCH"
GOOGLE_KEY_ENV = "GOOGLE_API_KEY"
GOOGLE_CX_ENV = "GOOGLE_CSE_ID"
ENGINE_GOOGLE = "google"
ENGINE_DUCKDUCKGO = "duckduckgo"
ENGINES = (ENGINE_GOOGLE, ENGINE_DUCKDUCKGO)
GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"
DUCKDUCKGO_SEARCH_URL = "https://html.duckduckgo.com/html/"
DUCKDUCKGO_LOCALE = "us-en"
DDGS_BACKEND = "auto"
IMPERSONATE = "chrome"
READER_ENV = "SUPERCLAW_READER"
OUTPUT_TOKENS_ENV = "SUPERCLAW_MAX_OUTPUT_TOKENS"
REDIRECT_CODES = (301, 302, 303, 307, 308)
BLOCKED_CODES = (401, 403, 429, 503)
FACT_KIND = "fact"
FILE_KIND = "file"
DOCUMENT_KIND = "document"
CHUNK_KIND = "chunk"
MAINT_KIND = "maintenance"
MAINT_NODE = "maint:last"
UNSAFE_SNAPSHOT = "unsafe"
INGESTED_EDGE = "ingested_in"
ANSWER_KINDS = ("fact", "obs", "chunk")
FROM_EDGE = "from"
SUPERSEDES_EDGE = "supersedes"
LEARNED_EDGE = "learned_in"
PRODUCED_EDGE = "produced"
READ_EDGE = "read"
WROTE_EDGE = "wrote"
TOUCH_VERBS = {"read_file": READ_EDGE, "edit_file": WROTE_EDGE, "write_file": WROTE_EDGE}
ORIGIN_WEB = "web"
YTDLP_BIN = "yt-dlp"
AUDIO_FORMAT = "mp3"
RG_BIN = "rg"
HOST_TOOLS = (("rg", "grep"), ("fd", "find"), ("sd", "sed"), ("xh", "curl"), ("jq", ""), ("uv", "pip"), ("bat", ""), ("eza", "ls"),
              ("delta", ""), ("hyperfine", "time"), ("gh", ""), (YTDLP_BIN, ""), ("ffmpeg", ""))
LOCALE_ENVS = ("LC_ALL", "LC_CTYPE", "LANG")
UTF8_MARK = "utf"
THOUSAND = 1_000
MILLION = 1_000_000
PRICE_UNIT_TOKENS = MILLION
PROC_DIR = "/proc"
STOPPED_STATES = ("T", "t")
PROMPTS_DIR = Path(__file__).parent / "prompts"
RECOMMENDED_MARK = "(recommended)"
PYTHON_SUFFIX = ".py"


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


def choose_engine(e: Mapping[str, str]) -> str:
    wanted = e.get(SEARCH_ENV, "").strip().lower()
    if wanted in ENGINES:
        return wanted
    return ENGINE_GOOGLE if e.get(GOOGLE_KEY_ENV, "").strip() and e.get(GOOGLE_CX_ENV, "").strip() else ENGINE_DUCKDUCKGO


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
PRICED_FILE = "priced.json"
WORKTREES_DIR = "worktrees"
CLIPBOARD_DIR = "clipboard"
MCP_FILE = "mcp.json"
MCP_SCOPES = ("user", "project")
WORKSPACE_DIR = ".superclaw"
AGENTS_DIR = "agents"
BUILTIN_PROFILES = Path(__file__).parent / "profiles"
COMMANDS_DIR = "commands"
REPO_MAP_IGNORED_DIRS = frozenset({".git", ".cache", ".next", ".worktrees", ".superclaw", "build", "coverage", "dist", "node_modules",
                                   "vendor", ".venv", "venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache", "target"})
SPECS_DIR = "specs"
SHARE_DIR = "superclaw"
PLUGINS_DIR = "plugins"
MARKETPLACES_DIR = "marketplaces"
MARKETPLACE_MANIFEST = ".claude-plugin/marketplace.json"
GITHUB_URL = "https://github.com/{repo}.git"
MARKETPLACE_SEP = "@"
PLUGIN_MANIFEST = "plugin.json"
PLUGIN_PARTS = ("skills", "agents", "commands", "hooks.json", MCP_FILE)
CLAUDE_PLUGIN_MANIFEST = ".claude-plugin/plugin.json"
CLAUDE_HOOKS_FILE = "hooks/hooks.json"
CLAUDE_MCP_FILE = ".mcp.json"
CLAUDE_DIR = ".claude"
CLAUDE_DIR_ENV = "CLAUDE_CONFIG_DIR"
CLAUDE_CONFIG_ENV = "SUPERCLAW_CLAUDE_CONFIG"
CLAUDE_INSTALLED_FILE = "plugins/installed_plugins.json"
CLAUDE_STATE_FILE = ".claude.json"
CLAUDE_SETTINGS_FILES = ("settings.json", "settings.local.json")
CLAUDE_GUIDELINES = ("CLAUDE.md", ".claude/CLAUDE.md")
PLUGIN_ROOT_VARS = ("CLAUDE_PLUGIN_ROOT", "SUPERCLAW_PLUGIN_ROOT")
FORMAT_SUPERCLAW = "superclaw"
FORMAT_CLAUDE = "claude"
HOOK_SHELL = "/bin/sh"
SESSION_END_OTHER = "other"
SESSION_END_EXIT = "prompt_input_exit"
SESSION_END_CLEAR = "clear"
PACKAGE = "supergraphdb"
PYPI_URL = f"https://pypi.org/pypi/{PACKAGE}/json"
UV_TOOLS_MARKER = "/uv/tools/"


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


def split_models(value: str) -> tuple[str, ...]:
    return tuple(part for part in (p.strip() for p in value.replace(",", " ").split()) if part)


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
    max_output_tokens_fallback: int = 8192
    message_overhead_tokens: int = 4
    completion_max_tokens: int = 32_768
    min_output_tokens: int = 1024
    completion_timeout_s: int = 120
    max_turns: int = 0
    usd_decimals: int = 6

    compaction_reserve_tokens: int = 16_384
    compaction_trigger_share: float = 0.6
    compaction_flush_calls: int = 6
    cache_hit_floor: float = 0.2
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
    ask_options_min: int = 3
    ask_options_max: int = 5
    intent_items_max: int = 8
    intent_goal_chars: int = 240
    intent_min_chars: int = 12
    failure_hint_at: int = 2
    failure_stop_at: int = 6
    stale_plan_tool_calls: int = 10
    tool_only_reminder_at: int = 6
    max_continue_nudges: int = 3
    identical_call_at: int = 3
    max_calls_per_turn: int = 42

    hook_timeout_s: int = 60
    hook_block_exit_code: int = 2
    mcp_connect_timeout_s: float = 8.0
    mcp_call_timeout_s: float = 120.0
    mcp_message_bytes: int = 4 * 1024 * 1024
    mcp_shutdown_wait_s: float = 0.5
    memory_recall_limit: int = 5
    guideline_file_bytes: int = 8 * 1024
    guideline_total_bytes: int = 32 * 1024
    skills_index_bytes: int = 8192
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

    review_diff_tokens: int = 24_000
    share_timeout_s: float = 300.0
    share_attach_timeout_s: float = 10.0
    share_attach_poll_s: float = 0.05
    share_key_chars: int = 12
    share_backlog: int = 16
    share_message_bytes: int = 16 * 1024 * 1024
    cron_tick_s: float = 30.0
    cron_id_chars: int = 40
    spec_slug_chars: int = 60
    spec_collisions: int = 1000
    verify_timeout_s: int = 120
    verify_output_lines: int = 8
    verify_attempts: int = 1
    verify_max_attempts: int = 5
    repo_map_files: int = 2000
    repo_map_depth: int = 6
    repo_map_bytes: int = 6000
    repo_map_matches: int = 20
    attachment_image_bytes: int = 5 * 1024 * 1024
    paste_lines_threshold: int = 3
    paste_chars_threshold: int = 2000
    drop_paths_max: int = 10
    clipboard_timeout_s: float = 5.0
    clipboard_keep: int = 50
    image_tokens: int = 1500
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
    web_search_results: int = 5
    web_search_results_max: int = 10
    web_search_timeout_s: float = 10.0
    web_search_body_bytes: int = 256 * 1024
    web_search_snippet_chars: int = 300
    web_search_min_interval_s: float = 1.5
    web_search_retry_s: float = 3.0
    web_search_attempts: int = 2
    web_fetch_bytes: int = 256 * 1024
    web_fetch_bytes_max: int = 2 * 1024 * 1024
    web_fetch_timeout_s: float = 30.0
    web_fetch_redirects: int = 5
    web_fetch_sniff_chars: int = 512
    web_fetch_outline_lines: int = 12
    web_raw_ttl_days: int = 7
    fact_recall_limit: int = 8
    fact_expand_seeds: int = 3
    fact_expand_depth: int = 2
    fact_expand_limit: int = 8
    answer_tokens: int = 1200
    ingest_ttl_days: int = 30
    fact_decay_days: int = 90
    fact_decay_factor: float = 0.5
    fact_confidence_floor: float = 0.2
    maintain_batch: int = 500
    maintain_stale_days: int = 1
    doc_search_limit: int = 6
    chunk_preview_chars: int = 200
    recall_path_sessions: int = 5
    recall_path_refs: int = 8
    fact_list_limit: int = 20
    facts_per_page_max: int = 12
    web_fact_confidence: float = 0.7
    days_per_month: int = 30
    days_per_year: int = 365
    download_bytes_max: int = 256 * 1024 * 1024
    download_timeout_s: float = 600.0
    download_chunk_bytes: int = 64 * 1024
    grep_timeout_s: float = 30.0
    shell_timeout_ms: int = 60_000
    shell_max_timeout_ms: int = 600_000
    shell_capture_bytes: int = 1024 * 1024
    shell_drain_timeout_s: float = 2.0
    shell_poll_s: float = 0.1
    capture_preview_lines: int = 5
    kernel_trace_depth: int = 3
    kernel_checkpoint_timeout_s: float = 10.0
    delegate_refs_returned: int = 5
    delegate_depth: int = 2
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
    session_search_limit: int = 10
    session_search_preview_chars: int = 120
    recent_sessions_shown: int = 20
    recent_models_shown: int = 5
    models_fetch_timeout_s: float = 15.0
    models_fetch_bytes: int = 8 * 1024 * 1024
    models_cache_ttl_s: int = 86_400
    worktree_name_chars: int = 80
    worktree_key_hash_chars: int = 10
    model_id_width: int = 44
    model_list_shown: int = 40
    replay_events_shown: int = 200
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
    palette_usage_width: int = 32
    palette_help_chars: int = 60
    status_near_compaction: float = 0.5
    resume_per_project: int = 12
    session_title_chars: int = 72
    tui_tier_narrow: int = 58
    tui_tier_medium: int = 80
    tui_tier_full: int = 100
    welcome_max_gap: int = 8


LIMITS = Limits()


@dataclass(frozen=True)
class Settings:
    model: str
    fallback_models: tuple[str, ...]
    mode: str
    effort: str
    stream: bool
    repo_map: bool
    search_engine: str
    google_search_key: str
    google_search_cx: str
    reader_url: str
    claude_config: bool
    claude_dir: Path
    claude_state: Path
    context_window: int
    output_tokens: int
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
        claude_dir = Path(e.get(CLAUDE_DIR_ENV, "").strip() or home / CLAUDE_DIR)
        return cls(
            model=e.get("SUPERCLAW_MODEL", "").strip() or DEFAULT_MODEL,
            fallback_models=split_models(e.get("SUPERCLAW_FALLBACK_MODELS", "")),
            mode=e.get("SUPERCLAW_MODE", "").strip() or DEFAULT_MODE,
            effort=e.get("SUPERCLAW_EFFORT", "").strip().lower() if e.get("SUPERCLAW_EFFORT", "").strip().lower() in EFFORTS else "",
            stream=e.get("SUPERCLAW_STREAM", "1").strip().lower() not in OFF_VALUES,
            repo_map=e.get("SUPERCLAW_REPO_MAP", "1").strip().lower() not in OFF_VALUES,
            search_engine=choose_engine(e),
            google_search_key=e.get(GOOGLE_KEY_ENV, "").strip(),
            google_search_cx=e.get(GOOGLE_CX_ENV, "").strip(),
            reader_url=e.get(READER_ENV, "").strip(),
            claude_config=e.get(CLAUDE_CONFIG_ENV, "").strip().lower() not in OFF_VALUES,
            claude_dir=claude_dir,
            claude_state=(claude_dir if e.get(CLAUDE_DIR_ENV, "").strip() else home) / CLAUDE_STATE_FILE,
            context_window=int(e.get("SUPERCLAW_CONTEXT_WINDOW", "").strip() or 0),
            output_tokens=int(e.get(OUTPUT_TOKENS_ENV, "").strip() or 0),
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
    def user_mcp(self) -> Path:
        return self.config_dir / MCP_FILE

    @property
    def credentials(self) -> Path:
        return self.config_dir / CREDENTIALS_FILE

    @property
    def models_cache(self) -> Path:
        return self.cache_dir / MODELS_CACHE_DIR

    @property
    def worktrees_dir(self) -> Path:
        return self.data_dir / WORKTREES_DIR

    @property
    def clipboard_dir(self) -> Path:
        return self.data_dir / CLIPBOARD_DIR

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

    def model_info(self) -> ModelInfo:
        from superclaw.models import lookup

        return lookup(self.model, self.models_cache)

    def window(self) -> int:
        return self.context_window or self.model_info().context_window

    def output_cap(self) -> int:
        return self.output_tokens or min(self.model_info().max_output_tokens, LIMITS.completion_max_tokens)

    @property
    def user_plugins(self) -> Path:
        return self.config_dir / PLUGINS_DIR

    @property
    def user_marketplaces(self) -> Path:
        return self.config_dir / MARKETPLACES_DIR

    def plugin_roots(self, workspace: Path | None = None) -> list[Path]:
        roots = [self.user_plugins]
        if workspace is not None:
            roots.insert(0, Path(workspace) / WORKSPACE_DIR / PLUGINS_DIR)
        return roots

    def plugins(self, workspace: Path | None = None, trusted: bool = True) -> list[Any]:
        from superclaw.plugins import discover

        return discover(self, workspace, trusted)

    def plugin_dirs(self, workspace: Path | None = None, trusted: bool = True) -> list[Path]:
        return [plugin.path for plugin in self.plugins(workspace, trusted)]

    def claude_roots(self, kind: str, workspace: Path | None = None) -> list[Path]:
        if not self.claude_config:
            return []
        roots = [self.claude_dir / kind]
        if workspace is not None:
            roots.insert(0, Path(workspace) / CLAUDE_DIR / kind)
        return roots

    def claude_settings(self, workspace: Path | None = None, trusted: bool = True) -> list[Path]:
        if not self.claude_config:
            return []
        files = [self.claude_dir / CLAUDE_SETTINGS_FILES[0]]
        if workspace is not None and trusted:
            files += [Path(workspace) / CLAUDE_DIR / name for name in CLAUDE_SETTINGS_FILES]
        return files

    def skill_roots(self, workspace: Path | None = None) -> list[Path | tuple[Path, str]]:
        roots: list[Path | tuple[Path, str]] = [self.skills_dir] if self.skills_dir else []
        roots += [self.config_dir / "skills", Path.home() / ".agents" / "skills"]
        if workspace is not None:
            roots.append(Path(workspace) / WORKSPACE_DIR / "skills")
        roots += self.claude_roots("skills", workspace)
        return roots + [(plugin.skills, plugin.id) if plugin.format == FORMAT_CLAUDE else plugin.skills for plugin in self.plugins(workspace) if plugin.skills]

    def agent_roots(self, workspace: Path | None = None) -> list[Path]:
        roots = [self.config_dir / AGENTS_DIR]
        if workspace is not None:
            roots.insert(0, Path(workspace) / WORKSPACE_DIR / AGENTS_DIR)
        return roots + self.claude_roots(AGENTS_DIR, workspace) + [plugin.agents for plugin in self.plugins(workspace) if plugin.agents] + [BUILTIN_PROFILES]

    def command_roots(self, workspace: Path | None = None) -> list[Path]:
        roots = [self.config_dir / COMMANDS_DIR]
        if workspace is not None:
            roots.insert(0, Path(workspace) / WORKSPACE_DIR / COMMANDS_DIR)
        return roots + self.claude_roots(COMMANDS_DIR, workspace) + [plugin.commands for plugin in self.plugins(workspace) if plugin.commands]
