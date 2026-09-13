from __future__ import annotations

CSS = """
Screen { layers: base popup; }
#title { height: 1; background: $panel; padding: 0 1; }
#welcome { height: 1fr; content-align: center middle; text-align: center; color: $text-muted; }
#welcome .wordmark { color: $accent; text-style: bold; }
#transcript { height: 1fr; padding: 0 1; }
#transcript > .user { color: $accent; text-style: bold; margin: 1 0 0 0; }
#transcript > .note { color: $text-muted; }
#transcript > .error { color: $error; }
#transcript > Markdown { margin: 0 0 1 0; }
#transcript > .child { margin-left: 4; }
ToolCard { height: auto; margin: 0 0 1 0; border-left: thick $panel; padding: 0 1; }
ToolCard.running { border-left: thick $accent; }
ToolCard.failed { border-left: thick $error; }
ToolCard .head { color: $text; }
ToolCard .head .name { text-style: bold; }
ToolCard .body { color: $text-muted; }
ToolCard .more { color: $text-muted; text-style: italic; }
#working { height: 1; padding: 0 1; color: $accent; }
#working.hidden { display: none; }
#status { height: 1; background: $panel; color: $text-muted; padding: 0 1; }
#prompt { dock: bottom; }
#palette { layer: popup; dock: bottom; margin: 0 0 3 0; height: auto; max-height: 8; width: 60; background: $surface; border: tall $accent; }
#palette.hidden { display: none; }
#dialog { width: 80%; max-width: 100; height: auto; padding: 1 2; background: $surface; border: thick $accent; }
#dialog .title { text-style: bold; margin-bottom: 1; }
#dialog .args { color: $text-muted; max-height: 12; overflow-y: auto; }
#dialog .reason { margin: 1 0; }
#dialog .buttons { height: auto; }
#dialog Button { margin-right: 1; }
"""

ADD = "green"
DEL = "red"
MUTED = "dim"
ACCENT = "bold"
