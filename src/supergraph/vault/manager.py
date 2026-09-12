"""VaultManager: file I/O for markdown notes."""
import os
from pathlib import Path
from datetime import datetime, timezone

from supergraph.vault.parser import (
    parse_frontmatter, parse_sections, extract_wikilinks,
    title_to_slug, write_frontmatter, write_section as _write_section,
    yaml,
)


class VaultManager:
    """Manages markdown note files in a vault directory."""

    def __init__(self, vault_path: str | Path):
        self._path = Path(vault_path)
        self._path.mkdir(parents=True, exist_ok=True)
        # Cache the resolved absolute vault root so every _safe_resolve call
        # avoids repeated syscalls. Refresh lazily if the path changes.
        self._resolved_root = self._path.resolve()

    @property
    def path(self) -> Path:
        return self._path

    def _safe_resolve(self, title_or_slug: str) -> Path:
        """Resolve a user-supplied title/slug into a vault-contained .md path.

        Rejects any input containing path separators or parent-dir components
        so DSL callers cannot escape the vault root. Returns the resolved
        Path. The file may or may not exist; callers are expected to handle
        FileNotFoundError themselves.

        Raises:
            ValueError: if the input contains path traversal markers or if
                        the resolved path lies outside the vault root.
        """
        if not isinstance(title_or_slug, str):
            raise ValueError("vault name must be a string")
        if not title_or_slug or not title_or_slug.strip():
            raise ValueError("vault name cannot be empty")
        # Reject path-component separators and parent-dir traversal. We reject
        # by raw substring rather than a path parse because Pathlib normalizes
        # ``..`` differently across platforms (``a/..`` collapses to ``.`` on
        # POSIX) which would let an attacker sneak through after normalization.
        forbidden = ("/", "\\", "\x00")
        for ch in forbidden:
            if ch in title_or_slug:
                raise ValueError(
                    f"invalid vault name {title_or_slug!r}: contains {ch!r}"
                )
        # Reject parent-dir markers as whole path components. Allow literal
        # ``..`` inside a longer filename (e.g. "readme..bak") but not as a
        # standalone segment.
        stripped = title_or_slug.strip()
        if stripped in ("..", "."):
            raise ValueError(
                f"invalid vault name {title_or_slug!r}: reserved path component"
            )

        candidate = (self._path / f"{title_or_slug}.md").resolve()

        # Final containment check. Symlinks inside the vault that point
        # outside are rejected here (resolve follows symlinks). A symlink to
        # a file inside the vault root is allowed.
        try:
            candidate.relative_to(self._resolved_root)
        except ValueError:
            raise ValueError(
                f"invalid vault name {title_or_slug!r}: resolves outside vault"
            )
        return candidate

    def new(self, title: str, kind: str = "memory", tags: list[str] | None = None,
            agent: str | None = None, body: str = "", summary: str = "") -> str:
        """Create a new note file. Returns the slug."""
        slug = title_to_slug(title)
        # Slugification strips path-unsafe characters but may yield an empty
        # string for inputs like "/" or "...". Guard so we don't try to create
        # a dotfile at the vault root.
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
        """Read full note content. Accepts title or slug.

        Resolution order:
          1. slugified form of the input (``title_to_slug``)
          2. exact form of the input

        Both candidates pass through ``_safe_resolve`` so a malicious
        ``title_or_slug`` cannot escape the vault.
        """
        # Try slugified form first; slugification already strips path-unsafe
        # characters, so this path is always in-vault.
        slug = title_to_slug(title_or_slug)
        if slug:
            candidate = self._safe_resolve(slug)
            if candidate.exists():
                return candidate.read_text(encoding="utf-8")
        # Fall through to exact match, but guard against traversal.
        candidate = self._safe_resolve(title_or_slug)
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Note not found: {title_or_slug}")

    def write_section(self, title_or_slug: str, section: str, content: str) -> None:
        """Overwrite a section in a note."""
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
        """Append to a section in a note."""
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
        """Create or return today's daily note. Returns slug (YYYY-MM-DD)."""
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = self._path / f"{today}.md"

        if not file_path.exists():
            self.new(today, kind="daily", agent=agent, summary=f"Daily note for {today}")

        return today

    def archive(self, title_or_slug: str) -> None:
        """Archive a note (set status = archived)."""
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
        """List all .md files in vault. Returns slugs (without .md)."""
        return [f.stem for f in sorted(self._path.glob("*.md"))]

    def get_mtime(self, slug: str) -> float:
        """Get file modification time for a note.

        Returns 0.0 for both "not found" and "invalid name" so callers that
        use mtime as a cache key can treat both as "needs resync".
        """
        try:
            file_path = self._safe_resolve(slug)
        except ValueError:
            return 0.0
        if not file_path.exists():
            return 0.0
        return file_path.stat().st_mtime
