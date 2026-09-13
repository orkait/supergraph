from __future__ import annotations

from textual import events
from textual.widgets import Input

from superclaw.clips import chip_bounds


class Composer(Input):
    def replace(self, text: str, start: int, end: int) -> None:
        super().replace(text, *chip_bounds(self.value, start, end))

    def _on_paste(self, event: events.Paste) -> None:
        if event.text and self.app.take_paste(event.text):  # type: ignore[attr-defined]
            event.prevent_default()
            event.stop()

    def action_paste(self) -> None:
        self.app.paste_clipboard()  # type: ignore[attr-defined]
