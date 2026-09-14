from __future__ import annotations

from dataclasses import dataclass

from superclaw.runtime import Message, Usage, message_tokens
from superclaw.settings import LIMITS


@dataclass
class ContextMeter:
    window: int
    reserve: int = LIMITS.compaction_reserve_tokens
    anchor: int = 0
    since: int = 0

    def observe(self, usage: Usage) -> None:
        self.anchor = usage.input_tokens + usage.output_tokens
        self.since = 0

    def append(self, message: Message) -> None:
        self.since += message_tokens(message)

    def reset(self) -> None:
        self.anchor = 0
        self.since = 0

    def used(self, estimate: int) -> int:
        return self.anchor + self.since if self.anchor else estimate

    def limit(self) -> int:
        return min(self.window - bounded(self.reserve, self.window), int(self.window * LIMITS.compaction_trigger_share))

    def pressure(self, estimate: int) -> bool:
        return self.window > 0 and self.used(estimate) > self.limit()


def bounded(tokens: int, window: int) -> int:
    return min(tokens, int(window * LIMITS.compaction_window_share)) if window > 0 else tokens
