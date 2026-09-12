from pathlib import Path
from datetime import datetime, timezone

from supergraph.vault.parser import (
    parse_sections, title_to_slug, write_frontmatter, write_section as _write_section,
    yaml,
)


class VaultManager:

    def __init__(self, vault_path: str | Path):
        self._path = Path(vault_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._resolved_root = self._path.resolve()

    @property
    def path(self) -> Path:
        return self._path

    def _safe_resolve(self, title_or_slug: str) -> Path:
        if not isinstance(title_or_slug, str):
            raise ValueError("vault name must be a string")
        if not title_or_slug or not title_or_slug.strip():
            raise ValueError("vault name cannot be empty")
        forbidden = ("/", "\\", "\x00")
        for ch in forbidden:
            if ch in title_or_slug:
                raise ValueError(
                    f"invalid vault name {title_or_slug!r}: contains {ch!r}"
                )
        stripped = title_or_slug.strip()
        if stripped in ("..", "."):
            raise ValueError(
                f"invalid vault name {title_or_slug!r}: reserved path component"
            )

        candidate = (self._path / f"{title_or_slug}.md").resolve()

        try:
            candidate.relative_to(self._resolved_root)
        except ValueError:
            raise ValueError(
                f"invalid vault name {title_or_slug!r}: resolves outside vault"
            )
        return candidate

    def new(self, title: str, kind: str = "memory", tags: list[str] | None = None,
            agent: str | None = None, body: str = "", summary: str = "") -> str:
        slug = title_to_slug(title)
        if not slug:
            raise ValueError(f"invalid note title: {title!r}")
        file_path = self._safe_resolve(slug)

        if file_path.exists():
            raise FileExistsError(f"Note already exists: {slug}.md")

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        fm = {
            "kind": kind,
            "tags": tags or [],
            "created": now,
            "updated": now,
            "status": "active",
        }
        if agent:
            fm["agent"] = agent

        fm_str = "---\n"
        fm_str += yaml.dump(fm, default_flow_style=False, sort_keys=False).strip()
        fm_str += "\n---\n"

        content = fm_str
        content += f"\n## Summary\n{summary or 'No summary yet.'}\n"
        content += f"\n## Body\n{body}\n"
        content += "\n## Links\n"

        file_path.write_text(content, encoding="utf-8")
        return slug

    def read(self, title_or_slug: str) -> str:
        slug = title_to_slug(title_or_slug)
        if slug:
            candidate = self._safe_resolve(slug)
            if candidate.exists():
                return candidate.read_text(encoding="utf-8")
        candidate = self._safe_resolve(title_or_slug)
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Note not found: {title_or_slug}")

    def write_section(self, title_or_slug: str, section: str, content: str) -> None:
        slug = title_to_slug(title_or_slug)
        file_path = self._safe_resolve(slug) if slug else self._safe_resolve(title_or_slug)
        if not file_path.exists():
            raise FileNotFoundError(f"Note not found: {title_or_slug}")

        old_content = file_path.read_text(encoding="utf-8")
        new_content = _write_section(old_content, section, content)
        new_content = write_frontmatter(new_content, {
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")
        })
        file_path.write_text(new_content, encoding="utf-8")

    def append_section(self, title_or_slug: str, section: str, content: str) -> None:
        slug = title_to_slug(title_or_slug)
        file_path = self._safe_resolve(slug) if slug else self._safe_resolve(title_or_slug)
        if not file_path.exists():
            raise FileNotFoundError(f"Note not found: {title_or_slug}")

        old_content = file_path.read_text(encoding="utf-8")
        sections = parse_sections(old_content)
        existing = sections.get(section.lower(), "")
        new_section_content = f"{existing}\n{content}" if existing else content
        new_content = _write_section(old_content, section, new_section_content)
        new_content = write_frontmatter(new_content, {
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")
        })
        file_path.write_text(new_content, encoding="utf-8")

    def daily(self, agent: str | None = None) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = self._path / f"{today}.md"

        if not file_path.exists():
            self.new(today, kind="daily", agent=agent, summary=f"Daily note for {today}")

        return today

    def archive(self, title_or_slug: str) -> None:
        slug = title_to_slug(title_or_slug)
        file_path = self._safe_resolve(slug) if slug else self._safe_resolve(title_or_slug)
        if not file_path.exists():
            raise FileNotFoundError(f"Note not found: {title_or_slug}")

        content = file_path.read_text(encoding="utf-8")
        content = write_frontmatter(content, {
            "status": "archived",
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        file_path.write_text(content, encoding="utf-8")

    def list_files(self) -> list[str]:
        return [f.stem for f in sorted(self._path.glob("*.md"))]

    def get_mtime(self, slug: str) -> float:
        try:
            file_path = self._safe_resolve(slug)
        except ValueError:
            return 0.0
        if not file_path.exists():
            return 0.0
        return file_path.stat().st_mtime
