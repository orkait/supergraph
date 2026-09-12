from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from supergraph import SuperGraph
from supergraph.ingest.llm.resolve import build_provider_chain

from superclaw.loop import Options, Result, run
from superclaw.memory import Memory
from superclaw.policy import Mode, Policy
from superclaw.prompt import PromptInputs, build_system_prompt
from superclaw.provider import LitellmProvider
from superclaw.runtime import Provider
from superclaw.sandbox import Backend, detect
from superclaw.session import SessionStore
from superclaw.skills import default_roots, load_skills
from superclaw.tools import Registry
from superclaw.tools.ask import AskUser
from superclaw.tools.files import core_file_tools
from superclaw.tools.plan import UpdatePlan
from superclaw.tools.shell import Bash
from superclaw.tools.skill import SkillTool

DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-flash"
DEFAULT_CONTEXT_WINDOW = 128_000


class NoProviderKey(RuntimeError):
    pass


def default_db_path() -> Path:
    override = os.environ.get("SUPERCLAW_DB_PATH", "").strip()
    if override:
        return Path(override)
    base = Path(os.environ.get("XDG_DATA_HOME", "").strip() or Path.home() / ".local" / "share")
    return base / "superclaw" / "brain"


def user_guidelines_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", "").strip() or Path.home() / ".config")
    return base / "superclaw" / "SUPERCLAW.md"


@dataclass
class Runtime:
    gs: Any
    store: SessionStore
    memory: Memory
    registry: Registry
    policy: Policy
    provider: Provider
    workspace: Path
    model: str
    max_turns: int = 12
    context_window: int = DEFAULT_CONTEXT_WINDOW

    @property
    def mode(self) -> Mode:
        return self.policy.mode

    @mode.setter
    def mode(self, value: Mode) -> None:
        self.policy.mode = value

    def close(self) -> None:
        self.gs.close()


def build_registry(memory: Memory, workspace: Path, backend: Backend | None = None) -> Registry:
    registry = Registry()
    for tool in (*core_file_tools(), Bash(backend), UpdatePlan(), SkillTool(roots=default_roots(workspace)), AskUser(),
                 memory.search_tool(), memory.note_tool()):
        registry.register(tool)
    return registry


def build_runtime(workspace: Path, mode: Mode, model: str, db_path: Path, **limits: int) -> Runtime:
    chain = build_provider_chain([model], free_first=False)
    if not chain:
        raise NoProviderKey(f"no API key resolved for model {model!r}; set the provider's key (for example OPENROUTER_API_KEY)")
    db_path.mkdir(parents=True, exist_ok=True)
    gs = SuperGraph(path=str(db_path))
    memory = Memory(gs)
    backend = detect()
    return Runtime(
        gs=gs, store=SessionStore(gs), memory=memory, registry=build_registry(memory, workspace, backend),
        policy=Policy(workspace, mode, sandboxed=backend is not None), provider=LitellmProvider(chain),
        workspace=workspace, model=model, **limits,
    )


def system_prompt_for(rt: Runtime, prompt: str) -> str:
    return build_system_prompt(PromptInputs(
        cwd=rt.workspace, mode=rt.mode, skills=load_skills(default_roots(rt.workspace)),
        memory=rt.memory.recall(prompt), user_guidelines=user_guidelines_path(),
        provider=rt.model.split("/", 1)[0], model=rt.model,
    ))


def resolve_session(rt: Runtime, resume: str | None) -> str:
    if not resume:
        return rt.store.create(cwd=str(rt.workspace), model=rt.model)
    sid = rt.store.latest() if resume == "latest" else resume
    if not sid or rt.store.get(sid) is None:
        raise KeyError(f"no session {resume!r}")
    return sid


def run_once(
    rt: Runtime,
    prompt: str,
    sid: str,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    on_permission: Callable[[dict[str, Any]], str] | None = None,
    on_ask_user: Callable[[list[dict[str, Any]]], list[str]] | None = None,
    require_completion: bool = False,
    verify: bool = False,
) -> Result:
    return run(prompt, rt.provider, Options(
        registry=rt.registry, policy=rt.policy, workspace=rt.workspace,
        system_prompt=system_prompt_for(rt, prompt), history=rt.store.replay(sid),
        max_turns=rt.max_turns, context_window=rt.context_window, require_completion_signal=require_completion, verify=verify,
        on_event=on_event, on_permission=on_permission, on_ask_user=on_ask_user,
        session=rt.store, session_id=sid,
    ))
