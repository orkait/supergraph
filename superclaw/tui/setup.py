from __future__ import annotations

from dataclasses import replace

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Static

from superclaw.app import Runtime, connect_provider
from superclaw.settings import PROVIDERS, Provider


class SetupScreen(ModalScreen[bool]):
    BINDINGS = [("escape", "later", "Later"), ("ctrl+c", "app.interrupt", "Quit")]

    def __init__(self, rt: Runtime) -> None:
        super().__init__()
        self.rt = rt
        self.provider: Provider = next((p for p in PROVIDERS if rt.model.startswith(p.name + "/")), PROVIDERS[0])

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Connect a provider", classes="title")
            yield Static("No API key was found. Pick a provider, paste its key, and superclaw stores it in ~/.config/superclaw/credentials.env (mode 600).", classes="reason")
            yield OptionList(*(f"{p.name:<12} {p.console}" for p in PROVIDERS), id="providers")
            yield Input(placeholder=f"{self.provider.env}", password=True, id="key")
            yield Input(value=self.rt.model, placeholder="model", id="model")
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
        self.query_one("#model", Input).value = self.provider.default_model

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
        model = self.query_one("#model", Input).value.strip() or self.provider.default_model
        if not key:
            self.query_one("#key", Input).focus()
            return
        self.rt.settings.save_credentials(self.provider, key, model)
        self.rt.settings = replace(self.rt.settings, model=model)
        self.rt.model = model
        self.rt.provider = connect_provider(model)
        self.dismiss(self.rt.provider is not None)

    def action_later(self) -> None:
        self.dismiss(False)
