from __future__ import annotations

import time
from dataclasses import dataclass, field

from rich.text import Text
from textual.widgets import Static

from superclaw.runtime import compact
from superclaw.settings import LIMITS
from superclaw.tui.theme import ACCENT, MUTED


@dataclass
class TurnTimer:
    started: float = 0.0
    paused_at: float = 0.0
    paused_for: float = 0.0
    calls: int = 0

    def start(self) -> None:
        self.started, self.paused_at, self.paused_for, self.calls = time.monotonic(), 0.0, 0.0, 0

    def pause(self) -> None:
        if self.started and not self.paused_at:
            self.paused_at = time.monotonic()

    def resume(self) -> None:
        if self.paused_at:
            self.paused_for += time.monotonic() - self.paused_at
            self.paused_at = 0.0

    def elapsed(self) -> float:
        if not self.started:
            return 0.0
        end = self.paused_at or time.monotonic()
        return end - self.started - self.paused_for


@dataclass
class RunStats:
    used: int = 0
    window: int = 0
    tokens: int = 0
    cost: float = 0.0
    saved: int = 0
    kept_out: int = 0
    cached: int = 0
    sent: int = 0
    timer: TurnTimer = field(default_factory=TurnTimer)

    @property
    def fill(self) -> float:
        return self.used / self.window if self.window else 0.0

    @property
    def cache_hit(self) -> float:
        return self.cached / self.sent if self.sent else 0.0


def tier(width: int) -> int:
    return sum(width >= bound for bound in (LIMITS.tui_tier_narrow, LIMITS.tui_tier_medium, LIMITS.tui_tier_full))


class TitleBar(Static):
    def on_resize(self) -> None:
        self.app.relayout()

    def show(self, cwd: str, branch: str, session: str, width: int) -> None:
        level = tier(width)
        left = Text(cwd, style=MUTED)
        if branch and level >= 1:
            left.append(f" {self.app.glyphs.dot} {branch}", style=MUTED)
        right = Text(session if level >= 3 else "", style=MUTED)
        gap = max(1, width - len(left) - len(right) - 2)
        self.update(left + Text(" " * gap) + right)


class StatusBar(Static):
    def show(self, mode: str, stats: RunStats, width: int) -> None:
        level = tier(width)
        glyphs = self.app.glyphs
        text = Text(f"{glyphs.mode} ", style=ACCENT)
        text.append(mode)
        if level >= 1 and self.app.rt.settings.effort:
            text.append(f" {glyphs.dot} {self.app.rt.settings.effort}", style=MUTED)
        if level >= 1 and stats.window:
            text.append(f"    {glyphs.gauge} {compact(stats.used)}/{compact(stats.window)} {glyphs.dot} {stats.fill:.1%}", style=MUTED)
        if level >= 2 and stats.cost:
            text.append(f"    ${stats.cost:.4f}", style=MUTED)
        if level >= 2 and (stats.saved or stats.kept_out):
            text.append(f"    kept out {compact(stats.saved + stats.kept_out)}", style=MUTED)
        self.update(text)


class WorkingLine(Static):
    def __init__(self) -> None:
        super().__init__(id="working", classes="hidden")
        self.phase = 0
        self.label = ""
        self.detail = ""

    def start(self, label: str, detail: str = "") -> None:
        self.label, self.detail = label, detail
        self.remove_class("hidden")

    def stop(self) -> None:
        self.add_class("hidden")

    def tick(self, elapsed: float, calls: int, tokens: int) -> None:
        glyphs = self.app.glyphs
        self.phase = (self.phase + 1) % len(glyphs.spinner)
        parts = [f"{elapsed:.0f}s", *([f"{calls} tools"] if calls else []), *([f"{compact(tokens)} tokens"] if tokens else [])]
        row = Text(f"{glyphs.spinner[self.phase]} {self.label}", style=ACCENT)
        if self.detail:
            row.append(f"  {self.detail}", style=MUTED)
        row.append(f"  {f' {glyphs.dot} '.join(parts)}", style=MUTED)
        row.truncate(max(1, self.content_size.width or self.app.size.width), overflow="ellipsis")
        self.update(row)
