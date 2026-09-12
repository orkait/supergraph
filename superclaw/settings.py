from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-flash"
DEFAULT_MODE = "ask"


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

    eager_schema_tokens: int = 1100
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

    id_hash_chars: int = 16
    run_id_bytes: int = 4
    session_id_bytes: int = 3
    event_seq_digits: int = 6
    session_list_limit: int = 1000
    session_events_limit: int = 100_000
    recent_sessions_shown: int = 20
    preview_args_chars: int = 160
    preview_error_chars: int = 200
    dialog_args_chars: int = 1200


LIMITS = Limits()


@dataclass(frozen=True)
class Settings:
    model: str
    mode: str
    context_window: int
    budget_tokens: int
    budget_usd: float
    data_dir: Path
    config_dir: Path
    db_path: Path
    skills_dir: Path | None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        home = Path.home()
        data_dir = Path(e.get("XDG_DATA_HOME", "").strip() or home / ".local" / "share") / "superclaw"
        config_dir = Path(e.get("XDG_CONFIG_HOME", "").strip() or home / ".config") / "superclaw"
        db_override = e.get("SUPERCLAW_DB_PATH", "").strip()
        skills_override = e.get("SUPERCLAW_SKILLS_DIR", "").strip()
        return cls(
            model=e.get("SUPERCLAW_MODEL", "").strip() or DEFAULT_MODEL,
            mode=e.get("SUPERCLAW_MODE", "").strip() or DEFAULT_MODE,
            context_window=int(e.get("SUPERCLAW_CONTEXT_WINDOW", "").strip() or 0),
            budget_tokens=int(e.get("SUPERCLAW_BUDGET_TOKENS", "").strip() or 0),
            budget_usd=float(e.get("SUPERCLAW_BUDGET_USD", "").strip() or 0),
            data_dir=data_dir,
            config_dir=config_dir,
            db_path=Path(db_override) if db_override else data_dir / "brain",
            skills_dir=Path(skills_override) if skills_override else None,
        )

    @property
    def user_guidelines(self) -> Path:
        return self.config_dir / "SUPERCLAW.md"

    @property
    def user_hooks(self) -> Path:
        return self.config_dir / "hooks.json"

    def model_info(self):
        from superclaw.models import lookup

        return lookup(self.model)

    def window(self) -> int:
        return self.context_window or self.model_info().context_window

    def skill_roots(self, workspace: Path | None = None) -> list[Path]:
        roots = [self.skills_dir] if self.skills_dir else []
        roots += [self.config_dir / "skills", Path.home() / ".agents" / "skills"]
        if workspace is not None:
            roots.append(Path(workspace) / ".superclaw" / "skills")
        return roots
