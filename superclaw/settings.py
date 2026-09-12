from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-flash"
DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_MODE = "ask"


@dataclass(frozen=True)
class Settings:
    model: str
    mode: str
    context_window: int
    budget_tokens: int
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
            context_window=int(e.get("SUPERCLAW_CONTEXT_WINDOW", "").strip() or DEFAULT_CONTEXT_WINDOW),
            budget_tokens=int(e.get("SUPERCLAW_BUDGET_TOKENS", "").strip() or 0),
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

    def skill_roots(self, workspace: Path | None = None) -> list[Path]:
        roots = [self.skills_dir] if self.skills_dir else []
        roots += [self.config_dir / "skills", Path.home() / ".agents" / "skills"]
        if workspace is not None:
            roots.append(Path(workspace) / ".superclaw" / "skills")
        return roots
