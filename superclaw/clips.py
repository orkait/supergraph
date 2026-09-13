from __future__ import annotations

import re
from dataclasses import dataclass, field

MARKER = re.compile(r"\[(Image|File|Pasted text) #(\d+)(?: \+\d+ lines)?\]")


@dataclass(frozen=True)
class Clip:
    label: str
    text: str = ""
    image: str = ""

    @property
    def marker(self) -> str:
        return f"[{self.label}]"


@dataclass
class Selection:
    prompt: str
    images: list[str] = field(default_factory=list)


class Clips:
    def __init__(self) -> None:
        self._clips: dict[str, Clip] = {}
        self._counts: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._clips)

    def add(self, kind: str, text: str = "", image: str = "", extra: str = "") -> Clip:
        self._counts[kind] = self._counts.get(kind, 0) + 1
        clip = Clip(f"{kind} #{self._counts[kind]}{extra}", text, image)
        self._clips[self._key(clip.label)] = clip
        return clip

    def add_pasted(self, text: str, lines: int) -> Clip:
        placeholder = self.add("Pasted text", extra=f" +{lines} lines")
        clip = Clip(placeholder.label, f'<attachment path="{placeholder.label}">\n{text.rstrip()}\n</attachment>')
        self._clips[self._key(clip.label)] = clip
        return clip

    @staticmethod
    def _key(label: str) -> str:
        match = MARKER.match(f"[{label}]")
        return f"{match.group(1)} #{match.group(2)}" if match else label

    def present(self, typed: str) -> list[Clip]:
        seen: list[Clip] = []
        for match in MARKER.finditer(typed):
            clip = self._clips.get(f"{match.group(1)} #{match.group(2)}")
            if clip is not None and clip not in seen:
                seen.append(clip)
        return seen

    def select(self, typed: str) -> Selection:
        chosen = self.present(typed)
        blocks = [c.text for c in chosen if c.text]
        return Selection("\n\n".join([typed.strip(), *blocks]).strip(), [c.image for c in chosen if c.image])

    def clear(self) -> None:
        self._clips.clear()
