from __future__ import annotations

import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from superclaw.settings import LIMITS, REPO_MAP_IGNORED_DIRS, Glyphs
from superclaw.text import count
from superclaw.viz import bars

LANGUAGES = {".py": "python", ".pyi": "python", ".ts": "typescript", ".tsx": "typescript", ".js": "javascript", ".jsx": "javascript",
             ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".rb": "ruby", ".php": "php", ".c": "c", ".h": "c",
             ".cpp": "cpp", ".hpp": "cpp", ".cs": "csharp", ".swift": "swift", ".sh": "shell", ".md": "markdown", ".json": "json",
             ".toml": "toml", ".yaml": "yaml", ".yml": "yaml", ".html": "html", ".css": "css", ".sql": "sql", ".lark": "lark"}
IMPORTANT = ("agents.md", "superclaw.md", "readme.md", "contributing.md", "go.mod", "go.sum", "package.json", "cargo.toml",
             "pyproject.toml", "requirements.txt", "makefile", "dockerfile", "docker-compose.yml", "docker-compose.yaml")
CLIPPED = "\n...[clipped]"


@dataclass
class RepoMap:
    root: Path
    files: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def directories(self) -> int:
        return len({"/".join(parts[:i]) for f in self.files for parts in [f.split("/")] for i in range(1, len(parts))})

    @property
    def languages(self) -> list[tuple[str, int]]:
        counted = Counter(LANGUAGES.get(Path(f).suffix.lower(), "") for f in self.files)
        counted.pop("", None)
        return sorted(counted.items(), key=lambda item: (-item[1], item[0]))

    @property
    def extensions(self) -> list[tuple[str, int]]:
        counted = Counter(Path(f).suffix.lower() for f in self.files)
        counted.pop("", None)
        return sorted(counted.items(), key=lambda item: (-item[1], item[0]))

    @property
    def important(self) -> list[str]:
        ranked = [(IMPORTANT.index(Path(f).name.lower()), f) for f in self.files if Path(f).name.lower() in IMPORTANT]
        return [f for _, f in sorted(ranked)]


def _git_files(root: Path) -> list[str] | None:
    done = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root, capture_output=True, check=False)
    if done.returncode != 0:
        return None
    return [f for f in done.stdout.decode("utf-8", errors="replace").split("\0") if f]


def _walk_files(root: Path, max_depth: int) -> list[str]:
    found: list[str] = []
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for entry in entries:
            if entry.name in REPO_MAP_IGNORED_DIRS or entry.is_symlink():
                continue
            if entry.is_dir():
                if depth + 1 <= max_depth:
                    stack.append((entry, depth + 1))
            else:
                found.append(entry.relative_to(root).as_posix())
    return found


def scan(root: Path, max_files: int = LIMITS.repo_map_files, max_depth: int = LIMITS.repo_map_depth) -> RepoMap:
    root = Path(root).resolve()
    files = _git_files(root)
    if files is None:
        files = _walk_files(root, max_depth)
    files = sorted(f for f in files if f.count("/") <= max_depth and not any(part in REPO_MAP_IGNORED_DIRS for part in f.split("/")))
    return RepoMap(root, files[:max_files], truncated=len(files) > max_files)


def render(found: RepoMap, budget: int = LIMITS.repo_map_bytes) -> str:
    if budget <= 0:
        return ""
    lines = [f"Repo: {found.root.name}", f"Counts: files={len(found.files)} dirs={found.directories}"]
    if found.truncated:
        lines.append("Truncated: true")
    lines.append("Important files: " + (", ".join(found.important) or "none"))
    lines.append("Languages: " + (", ".join(f"{name}={count}" for name, count in found.languages) or "none"))
    lines.append("Extensions: " + (", ".join(f"{ext}={count}" for ext, count in found.extensions) or "none"))
    text = "\n".join(lines) + "\n\nFiles:\n" + "\n".join(f"  {f}" for f in found.files)
    if len(text.encode()) <= budget:
        return text
    cut = text.encode()[: max(0, budget - len(CLIPPED))].decode("utf-8", errors="ignore")
    return cut.rsplit("\n", 1)[0] + CLIPPED


def search(found: RepoMap, query: str, limit: int = LIMITS.repo_map_matches) -> list[tuple[str, str]]:
    terms = [t for t in query.lower().replace("/", " ").split() if t]
    if not terms:
        return []
    scored: list[tuple[int, str, str]] = []
    for path in found.files:
        low = path.lower()
        hits = [t for t in terms if t in low]
        if not hits:
            continue
        segment = any(t in low.split("/") or t in Path(low).stem.split("_") for t in hits)
        score = len(hits) * 10 + (5 if len(hits) == len(terms) else 0) + (3 if segment else 0) - min(len(path) // 20, 4)
        reason = "all terms" if len(hits) == len(terms) else f"{len(hits)} of {len(terms)} terms"
        scored.append((score, path, reason + (", path segment" if segment else "")))
    return [(path, reason) for _, path, reason in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]]


def chart(found: RepoMap, glyphs: Glyphs, width: int) -> list[str]:
    roots = Counter(path.split("/")[0] if "/" in path else "." for path in found.files)
    out = [f"{found.root.name}  {count(len(found.files), 'file')}  {count(found.directories, 'directory', 'directories')}"]
    for title, rows in (("languages", found.languages), ("top level", roots.most_common(LIMITS.chart_rows)), ("extensions", found.extensions)):
        shown = [(name, float(size)) for name, size in list(rows)[: LIMITS.chart_rows]]
        if shown:
            out += ["", title, *(f"  {line}" for line in bars(shown, width, glyphs))]
    return out
