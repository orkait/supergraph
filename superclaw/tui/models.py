from __future__ import annotations

import os

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from superclaw.app import Runtime
from superclaw.catalog import Model, describe, models_for
from superclaw.settings import LIMITS, Provider
from superclaw.tui.theme import ACCENT, MUTED

RECENT_GROUP = "recent"
FETCHING = "fetching live lists"


class ModelScreen(ModalScreen[str]):
    BINDINGS = [
        ("escape", "later", "Keep"),
        ("ctrl+c", "app.interrupt", "Quit"),
        ("down", "move(1)", "Next"),
        ("up", "move(-1)", "Previous"),
    ]

    def __init__(self, rt: Runtime, providers: list[Provider], active: str, recent: list[str]) -> None:
        super().__init__()
        self.rt = rt
        self.providers = providers
        self.active = active
        self.recent = recent
        self.models: dict[str, list[Model]] = {}
        self.rows: list[Model | None] = []
        self.pending = len(providers)

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Choose a model", classes="title")
            yield Input(placeholder="type to filter", id="filter")
            yield OptionList(id="models")
            yield Static("", id="hint", classes="reason")

    def on_mount(self) -> None:
        self.refill()
        self.query_one("#filter", Input).focus()
        self.run_worker(self.discover, thread=True, exclusive=True)

    def discover(self) -> None:
        for online in (False, True):
            for provider in self.providers:
                models = models_for(provider, os.environ.get(provider.env, ""), self.rt.settings.models_cache, online=online)
                self.app.call_from_thread(self.learned, provider.name, models, online)

    def learned(self, name: str, models: list[Model], final: bool) -> None:
        self.models[name] = models
        self.pending -= final
        self.refill()

    def refill(self) -> None:
        text = self.query_one("#filter", Input).value.strip().lower()
        options: list[Option] = []
        self.rows = []
        everything = [m for provider in self.providers for m in self.models.get(provider.name, [])]
        recent = [m for wanted in self.recent for m in everything if m.id == wanted]
        groups = [(RECENT_GROUP, recent), *((p.name, self.models.get(p.name, [])) for p in self.providers)]
        for title, models in groups:
            shown = [m for m in models if text in m.id.lower() or text in m.name.lower()]
            if not shown:
                continue
            options.append(Option(Text(title, style=MUTED), disabled=True))
            self.rows.append(None)
            for model in shown:
                options.append(Option(self.row(model)))
                self.rows.append(model)
        table = self.query_one("#models", OptionList)
        table.clear_options()
        table.add_options(options)
        first = next((i for i, m in enumerate(self.rows) if m is not None), None)
        table.highlighted = next((i for i, m in enumerate(self.rows) if m and m.id == self.active), first)
        dot = self.app.glyphs.dot
        status = FETCHING if self.pending else f"{len(everything)} models"
        self.query_one("#hint", Static).update(f" {dot} ".join(("enter picks", f"esc keeps {self.active}", "$ per M", status)))

    def row(self, model: Model) -> Text:
        head = Text(f"{model.id:<{LIMITS.model_id_width}}", style=ACCENT if model.id == self.active else "")
        return head + Text(f" {describe(model, self.app.glyphs.dot)}", style=MUTED)

    def on_input_changed(self, event: Input.Changed) -> None:
        event.stop()
        self.refill()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        index = self.query_one("#models", OptionList).highlighted
        if index is not None and self.rows[index] is not None:
            self.dismiss(self.rows[index].id)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        model = self.rows[event.option_index]
        if model is not None:
            self.dismiss(model.id)

    def action_move(self, step: int) -> None:
        table = self.query_one("#models", OptionList)
        table.action_cursor_down() if step > 0 else table.action_cursor_up()

    def action_later(self) -> None:
        self.dismiss("")
