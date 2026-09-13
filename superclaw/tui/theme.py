from __future__ import annotations

CSS = """
Screen { layers: base popup; }
#title { height: 1; padding: 0 1; color: $text-muted; }
#welcome { height: 1fr; content-align: center middle; text-align: center; color: $text-muted; }
#transcript { height: 1fr; padding: 0 1; }
#transcript > .user { color: $accent; text-style: bold; margin: 1 0 0 0; border-left: thick $accent; padding-left: 1; }
#transcript > .note { color: $text-muted; }
#transcript > .error { color: $error; }
#transcript > Markdown { margin: 0 0 1 0; }
#transcript > .child { margin-left: 4; }
ToolCard { height: auto; margin: 0 0 1 0; border-left: thick $panel; padding: 0 1; }
ToolCard.running { border-left: thick $accent; }
ToolCard.failed { border-left: thick $error; }
ToolCard .head { color: $text; }
ToolCard .body { color: $text-muted; }
ToolCard .more { color: $text-muted; text-style: italic; }
#hints { height: 1; padding: 0 2; color: $text-muted; }
#hints.hidden { display: none; }
#working { height: 1; padding: 0 2; color: $accent; }
#working.hidden { display: none; }
#composer { height: 3; border: $border-kind $panel; border-subtitle-color: $text-muted; padding: 0 1; }
#composer:focus-within { border: $border-kind $accent; }
#composer .gutter { width: 2; color: $accent; text-style: bold; }
#prompt { border: none; height: 1; padding: 0; background: transparent; }
#prompt:focus { border: none; background: transparent; }
#status { height: 1; padding: 0 2; color: $text-muted; }
#palette { layer: popup; dock: bottom; margin: 0 0 5 2; height: auto; max-height: 8; width: 64; background: $surface; border: $border-kind $panel; }
#palette.hidden { display: none; }
ModalScreen { align: center middle; }
#dialog { width: 80%; max-width: 100; height: auto; padding: 1 2; background: $surface; border: $border-kind $accent; }
#dialog .title { text-style: bold; margin-bottom: 1; }
#dialog .args { color: $text-muted; max-height: 12; overflow-y: auto; }
#dialog .reason { margin: 1 0; }
#dialog .buttons { height: auto; }
#dialog Button { margin-right: 1; }
#dialog #providers { height: auto; max-height: 8; margin-bottom: 1; }
#dialog #models { height: auto; max-height: 18; margin-bottom: 1; text-wrap: nowrap; text-overflow: ellipsis; }
#dialog Input { margin-bottom: 1; }
"""

ADD = "green"
DEL = "red"
ADD_ROW = "on #1d2a1d"
DEL_ROW = "on #2c1c1c"
MUTED = "dim"
ACCENT = "bold"
