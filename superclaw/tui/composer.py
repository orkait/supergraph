from __future__ import annotations

from textual import events
from textual.widgets import Input


class Composer(Input):
    def _on_paste(self, event: events.Paste) -> None:
        if event.text and self.app.take_paste(event.text):  # type: ignore[attr-defined]
            event.prevent_default()
            event.stop()

    def action_paste(self) -> None:
        self.app.paste_clipboard()  # type: ignore[attr-defined]
