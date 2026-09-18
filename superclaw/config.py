from __future__ import annotations

from dataclasses import dataclass, replace

from superclaw.app import Runtime, apply_effort, apply_sandbox, connect_provider, switch_model
from superclaw.policy import Mode
from superclaw.settings import (
    BUDGET_TOKENS_ENV,
    BUDGET_USD_ENV,
    CLAUDE_CONFIG_ENV,
    CONTEXT_WINDOW_ENV,
    EFFORT_ENV,
    EFFORT_OFF,
    EFFORTS,
    ENGINES,
    FALLBACK_ENV,
    INTENT_GATE_ENV,
    MAX_TURNS_ENV,
    MODE_ENV,
    MODEL_ENV,
    OUTPUT_TOKENS_ENV,
    RENDERER_ENV,
    RENDERERS,
    REPO_MAP_ENV,
    SANDBOX_ENV,
    SANDBOX_OFF,
    SANDBOX_ON,
    SEARCH_ENV,
    STREAM_ENV,
    Settings,
    split_models,
)

BOOL, CHOICE, NUMBER, TEXT = "bool", "choice", "number", "text"
NEXT_LAUNCH = "next launch"
UNSET = "-"
TRUE_WORDS = (SANDBOX_ON, "1", "true", "yes")
FALSE_WORDS = (SANDBOX_OFF, "0", "false", "no")


@dataclass(frozen=True)
class Option:
    key: str
    env: str
    kind: str
    label: str
    choices: tuple[str, ...] = ()
    live: bool = True


OPTIONS = (
    Option("model", MODEL_ENV, TEXT, "the model every turn runs on"),
    Option("fallback", FALLBACK_ENV, TEXT, "models to try when the first one fails, comma separated"),
    Option("effort", EFFORT_ENV, CHOICE, "how hard the model thinks per turn", (*EFFORTS, EFFORT_OFF)),
    Option("mode", MODE_ENV, CHOICE, "what runs without asking you first", tuple(m.value for m in Mode)),
    Option("sandbox", SANDBOX_ENV, BOOL, "run bash and the python kernel inside bubblewrap"),
    Option("repo_map", REPO_MAP_ENV, BOOL, "put a map of the repository in every system prompt"),
    Option("stream", STREAM_ENV, BOOL, "show the answer as it is generated"),
    Option("intent_gate", INTENT_GATE_ENV, BOOL, "classify the request before acting, and refuse to guess"),
    Option("max_turns", MAX_TURNS_ENV, NUMBER, "model turns one prompt may take, 0 for no cap"),
    Option("budget_tokens", BUDGET_TOKENS_ENV, NUMBER, "tokens one prompt may spend, 0 for no cap"),
    Option("budget_usd", BUDGET_USD_ENV, NUMBER, "dollars one prompt may spend, 0 for no cap"),
    Option("context_window", CONTEXT_WINDOW_ENV, NUMBER, "override the model's window, 0 to use the catalog's"),
    Option("output_tokens", OUTPUT_TOKENS_ENV, NUMBER, "cap the model's reply, 0 to use the model's own"),
    Option("tui", RENDERER_ENV, CHOICE, "keeps scrollback, or takes the whole screen", RENDERERS, live=False),
    Option("search", SEARCH_ENV, CHOICE, "engine behind web_search", ENGINES, live=False),
    Option("claude_config", CLAUDE_CONFIG_ENV, BOOL, "also read ~/.claude skills, agents and servers", live=False),
)


def find(key: str) -> Option | None:
    return next((o for o in OPTIONS if o.key == key), None)


def current(settings: Settings, option: Option) -> str:
    if option.key == "fallback":
        return ", ".join(settings.fallback_models) or UNSET
    value = getattr(settings, {"tui": "renderer", "search": "search_engine"}.get(option.key, option.key))
    if option.kind == BOOL:
        return SANDBOX_ON if value else SANDBOX_OFF
    if option.kind == NUMBER:
        return UNSET if not value else f"{value:g}" if isinstance(value, float) else str(value)
    return str(value) or UNSET


def normalize(option: Option, raw: str) -> str:
    text = raw.strip()
    if option.kind == BOOL:
        low = text.lower()
        if low in TRUE_WORDS:
            return SANDBOX_ON
        if low in FALSE_WORDS:
            return SANDBOX_OFF
        raise ValueError(f"{option.key} takes {SANDBOX_ON} or {SANDBOX_OFF}")
    if option.kind == CHOICE:
        if text.lower() not in option.choices:
            raise ValueError(f"{option.key} takes {' | '.join(option.choices)}")
        return text.lower()
    if option.kind == NUMBER:
        try:
            number = float(text or 0)
        except ValueError:
            raise ValueError(f"{option.key} takes a number") from None
        if number < 0:
            raise ValueError(f"{option.key} cannot be negative")
        return f"{number:g}"
    return text


def apply(rt: Runtime, option: Option, value: str) -> str:
    if option.key == "model":
        switch_model(rt, value)
        return f"model {rt.model}"
    if option.key == "sandbox":
        return apply_sandbox(rt, value == SANDBOX_ON)
    if option.key == "effort":
        apply_effort(rt, "" if value == EFFORT_OFF else value)
        return f"effort {value}"
    rt.settings.save_option(option.env, value)
    rt.settings = replace(rt.settings, **_field(option, value))
    if option.key == "mode":
        rt.mode = Mode(value)
    elif option.key == "max_turns":
        rt.max_turns = rt.settings.max_turns
    elif option.key == "budget_tokens":
        rt.token_budget = rt.settings.budget_tokens
    elif option.key == "intent_gate":
        rt.intent_gate = rt.settings.intent_gate
    elif option.key in ("fallback", "output_tokens", "stream"):
        rt.provider = connect_provider(rt.model, rt.settings.effort, rt.settings.fallback_models, rt.settings.stream, rt.settings.output_cap())
    return f"{option.key} {value}" + ("" if option.live else f", from the {NEXT_LAUNCH}")


def _field(option: Option, value: str) -> dict[str, object]:
    name = {"tui": "renderer", "search": "search_engine", "fallback": "fallback_models"}.get(option.key, option.key)
    if option.kind == BOOL:
        return {name: value == SANDBOX_ON}
    if option.key == "fallback":
        return {name: split_models(value)}
    if option.kind == NUMBER:
        return {name: float(value) if option.key == "budget_usd" else int(float(value))}
    return {name: value}


def set_option(rt: Runtime, key: str, raw: str) -> str:
    option = find(key)
    if option is None:
        raise KeyError(f"unknown option {key!r}; try one of {', '.join(o.key for o in OPTIONS)}")
    try:
        return apply(rt, option, normalize(option, raw))
    except ValueError as e:
        raise KeyError(str(e)) from None


def lines(settings: Settings) -> list[tuple[Option, str]]:
    return [(option, current(settings, option)) for option in OPTIONS]
