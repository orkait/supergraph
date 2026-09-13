from __future__ import annotations

import time
from dataclasses import dataclass, field

from rich.text import Text
from textual.widgets import Static

from superclaw.settings import LIMITS
from superclaw.tui.theme import ACCENT, MUTED

WORDMARK = "superclaw"
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SEPARATOR = "  │  "
GAUGE_FULL = "▰"
GAUGE_EMPTY = "▱"


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
    timer: TurnTimer = field(default_factory=TurnTimer)

    @property
    def fill(self) -> float:
        return self.used / self.window if self.window else 0.0


def gauge(fill: float) -> str:
    full = min(LIMITS.gauge_cells, round(fill * LIMITS.gauge_cells))
    return GAUGE_FULL * full + GAUGE_EMPTY * (LIMITS.gauge_cells - full)


def tier(width: int) -> int:
    return sum(width >= bound for bound in (LIMITS.tui_tier_narrow, LIMITS.tui_tier_medium, LIMITS.tui_tier_full))


class TitleBar(Static):
    def show(self, cwd: str, branch: str, model: str, session: str, width: int) -> None:
        level = tier(width)
        left = Text(WORDMARK, style=ACCENT)
        left.append(f"  {cwd}", style=MUTED)
        if branch and level >= 1:
            left.append(f" · {branch}", style=MUTED)
        right = Text(model if level >= 2 else "", style=MUTED)
        if level >= 3:
            right.append(f" · {session}", style=MUTED)
        gap = max(1, width - len(left) - len(right) - 2)
        self.update(left + Text(" " * gap) + right)


class StatusBar(Static):
    def show(self, mode: str, stats: RunStats, width: int) -> None:
        level = tier(width)
        text = Text("● ", style=ACCENT)
        text.append(mode)
        if level >= 1:
            text.append(SEPARATOR + gauge(stats.fill) + f" {stats.fill:.1%} of {stats.window:,}" if stats.window else SEPARATOR + f"{stats.tokens:,} tokens", style=MUTED)
        if level >= 2:
            text.append(SEPARATOR + f"${stats.cost:.4f}", style=MUTED)
        self.update(text)


class WorkingLine(Static):
    def __init__(self) -> None:
        super().__init__(id="working", classes="hidden")
        self.phase = 0
        self.label = ""

    def start(self, label: str) -> None:
        self.label = label
        self.remove_class("hidden")

    def stop(self) -> None:
        self.add_class("hidden")

    def tick(self, elapsed: float, calls: int) -> None:
        self.phase = (self.phase + 1) % len(SPINNER)
        detail = f"  {elapsed:.0f}s" + (f" · {calls} tools" if calls else "")
        self.update(Text(f"{SPINNER[self.phase]} {self.label}", style=ACCENT) + Text(detail, style=MUTED))
