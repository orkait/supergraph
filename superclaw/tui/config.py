from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList, Static

from superclaw.app import Runtime
from superclaw.config import BOOL, CHOICE, NEXT_LAUNCH, OPTIONS, Option, current, set_option
from superclaw.settings import LIMITS, SANDBOX_OFF, SANDBOX_ON
from superclaw.text import clip
from superclaw.tui.theme import ACCENT, MUTED

KEY_WIDTH = 16
VALUE_WIDTH = 26
MODEL_KEY = "model"


def cycled(option: Option, value: str) -> str:
    choices = (SANDBOX_ON, SANDBOX_OFF) if option.kind == BOOL else option.choices
    return choices[(choices.index(value) + 1) % len(choices)] if value in choices else choices[0]


class ConfigScreen(ModalScreen[None]):
    BINDINGS = [
        ("escape", "close", "Close"),
        ("ctrl+c", "app.interrupt", "Quit"),
        ("down", "move(1)", "Next"),
        ("up", "move(-1)", "Previous"),
    ]

    def __init__(self, rt: Runtime) -> None:
        super().__init__()
        self.rt = rt

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Settings", classes="title")
            yield OptionList(id="settings")
            yield Input(placeholder="type a value for the highlighted setting, or press enter on it to cycle", id="value")
            yield Static("", id="note", classes="reason")

    def on_mount(self) -> None:
        self.refill()
        self.query_one("#settings", OptionList).focus()

    def refill(self, note: str = "", error: bool = False) -> None:
        table = self.query_one("#settings", OptionList)
        keep = table.highlighted
        table.clear_options()
        for option in OPTIONS:
            table.add_option(self.row(option))
        table.highlighted = keep if keep is not None else 0
        dot = self.app.glyphs.dot
        hint = f"enter cycles or edits {dot} esc closes {dot} saved to {self.rt.settings.credentials}"
        self.query_one("#note", Static).update(note or hint)
        self.query_one("#note", Static).set_class(error, "error")

    def row(self, option: Option) -> Text:
        value = current(self.rt.settings, option)
        label = option.label if option.live else f"{option.label} ({NEXT_LAUNCH})"
        return (Text(f"{option.key:<{KEY_WIDTH}}", style=ACCENT)
                + Text(f"{clip(value, VALUE_WIDTH):<{VALUE_WIDTH}} ", style="")
                + Text(clip(label, LIMITS.palette_help_chars), style=MUTED))

    def highlighted(self) -> Option:
        index = self.query_one("#settings", OptionList).highlighted
        return OPTIONS[index or 0]

    def set_value(self, option: Option, value: str) -> None:
        try:
            told = set_option(self.rt, option.key, value)
        except KeyError as e:
            self.refill(str(e.args[0]), error=True)
            return
        self.app.refresh_status()
        self.refill(told)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        option = OPTIONS[event.option_index]
        if option.key == MODEL_KEY:
            self.app.open_models()
        elif option.kind in (BOOL, CHOICE):
            self.set_value(option, cycled(option, current(self.rt.settings, option)))
        else:
            field = self.query_one("#value", Input)
            field.value = current(self.rt.settings, option)
            field.focus()

    def on_screen_resume(self) -> None:
        self.refill()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        field = self.query_one("#value", Input)
        value, field.value = field.value, ""
        self.set_value(self.highlighted(), value)
        self.query_one("#settings", OptionList).focus()

    def action_move(self, step: int) -> None:
        table = self.query_one("#settings", OptionList)
        table.action_cursor_down() if step > 0 else table.action_cursor_up()

    def action_close(self) -> None:
        self.dismiss(None)
