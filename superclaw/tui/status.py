from __future__ import annotations

import time
from dataclasses import dataclass, field

from rich.text import Text
from textual.widgets import Static

from superclaw.settings import LIMITS
from superclaw.text import compact
from superclaw.tui.theme import ACCENT, MUTED
from superclaw.viz import gauge


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


CYCLE_HINT = "shift+tab to cycle"


@dataclass(frozen=True)
class Where:
    path: str
    branch: str = ""
    model: str = ""
    effort: str = ""
    sandbox: str = ""
    agent: str = ""
    session: str = ""


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
    limit: int = 0
    timer: TurnTimer = field(default_factory=TurnTimer)

    @property
    def fill(self) -> float:
        return self.used / self.window if self.window else 0.0

    @property
    def cache_hit(self) -> float:
        return self.cached / self.sent if self.sent else 0.0


def tier(width: int) -> int:
    return sum(width >= bound for bound in (LIMITS.tui_tier_narrow, LIMITS.tui_tier_medium, LIMITS.tui_tier_full))


class StatusBar(Static):
    def on_resize(self) -> None:
        self.app.relayout()

    def facts(self, where: Where, stats: RunStats, level: int) -> Text:
        glyphs = self.app.glyphs
        parts = [where.path]
        if where.branch and level >= 1:
            parts.append(where.branch)
        if stats.window and level >= 1:
            meter = gauge(stats.used, stats.window, LIMITS.status_gauge_width, glyphs) if level >= 2 else glyphs.gauge
            parts.append(f"{meter} {self.context(stats)}")
        if stats.cost and level >= 2:
            parts.append(f"${stats.cost:.4f}")
        if level >= 2:
            parts.append(where.model)
        if where.effort and level >= 3:
            parts.append(where.effort)
        return Text("  ".join(parts), style=MUTED)

    def context(self, stats: RunStats) -> str:
        used, limit = compact(stats.used), stats.limit
        if limit and stats.used > limit * LIMITS.status_near_compaction:
            return f"{used}/{compact(stats.window)} {self.app.glyphs.dot} {max(0.0, 1 - stats.used / limit):.0%} until compaction"
        return f"{used}/{compact(stats.window)} {self.app.glyphs.dot} {stats.fill:.0%}"

    def state(self, mode: str, where: Where, stats: RunStats, level: int) -> Text:
        glyphs = self.app.glyphs
        text = Text(f"{glyphs.mode} ", style=ACCENT)
        text.append(f"{mode} ({CYCLE_HINT})" if level >= 2 else mode)
        parts = []
        if where.sandbox and level >= 3:
            parts.append(f"sandbox {where.sandbox}")
        if stats.sent and level >= 2:
            parts.append(f"{stats.cache_hit:.0%} cached")
        if (stats.saved or stats.kept_out) and level >= 2:
            parts.append(f"kept out {compact(stats.saved + stats.kept_out)}")
        if where.agent and level >= 1:
            parts.append(f"agent {where.agent}")
        if where.session and level >= 3:
            parts.append(where.session)
        if parts:
            text.append(f" {glyphs.dot} " + f" {glyphs.dot} ".join(parts), style=MUTED)
        return text

    def show(self, mode: str, where: Where, stats: RunStats, width: int) -> None:
        level = tier(width)
        self.update(self.facts(where, stats, level) + Text("\n") + self.state(mode, where, stats, level))


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
