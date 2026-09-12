from __future__ import annotations

import re
from pathlib import Path

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


class SpillStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def save(self, session_id: str, tool: str, call_id: str, text: str) -> Path:
        folder = self.root / (_SAFE.sub("_", session_id) or "no-session")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{_SAFE.sub('_', tool)}-{_SAFE.sub('_', call_id) or 'call'}.txt"
        path.write_text(text)
        return path
