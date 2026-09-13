from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Static

from superclaw.app import Runtime, switch_model
from superclaw.catalog import provider_of
from superclaw.settings import PROVIDERS, Provider
from superclaw.tui.models import ModelScreen


class SetupScreen(ModalScreen[bool]):
    BINDINGS = [("escape", "later", "Later"), ("ctrl+c", "app.interrupt", "Quit")]

    def __init__(self, rt: Runtime, provider: Provider | None = None) -> None:
        super().__init__()
        self.rt = rt
        self.provider: Provider = provider or provider_of(rt.model) or PROVIDERS[0]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Connect a provider", classes="title")
            yield Static("Pick a provider and paste its key; superclaw stores it in ~/.config/superclaw/credentials.env (mode 600), then lists the models it serves.", classes="reason")
            yield OptionList(*(f"{p.name:<12} {p.console}" for p in PROVIDERS), id="providers")
            yield Input(placeholder=f"{self.provider.env}", password=True, id="key")
            with Horizontal(classes="buttons"):
                yield Button("Connect", id="connect", variant="primary")
                yield Button("Later (esc)", id="later")

    def on_mount(self) -> None:
        self.query_one("#providers", OptionList).highlighted = PROVIDERS.index(self.provider)
        self.call_after_refresh(self.query_one("#key", Input).focus)

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        event.stop()
        self.provider = PROVIDERS[event.option_index]
        self.query_one("#key", Input).placeholder = self.provider.env

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.query_one("#key", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.connect()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "connect":
            self.connect()
        else:
            self.dismiss(False)

    def connect(self) -> None:
        key = self.query_one("#key", Input).value.strip()
        if not key:
            self.query_one("#key", Input).focus()
            return
        current = self.rt.model if self.rt.model.startswith(self.provider.name + "/") else self.provider.default_model
        self.rt.settings.save_credentials(self.provider, key, current)
        self.app.push_screen(ModelScreen(self.rt, [self.provider], current, []), self.chosen)

    def chosen(self, model: str | None) -> None:
        switch_model(self.rt, model or (self.rt.model if provider_of(self.rt.model) is self.provider else self.provider.default_model))
        self.dismiss(self.rt.provider is not None)

    def action_later(self) -> None:
        self.dismiss(False)
