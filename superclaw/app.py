from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from supergraph import SuperGraph
from supergraph.ingest.llm.resolve import build_provider_chain

from superclaw.hooks import Dispatcher, load_hooks
from superclaw.intent import classify
from superclaw.loop import Options, Result, run
from superclaw.memory import Memory
from superclaw.policy import Mode, Policy
from superclaw.prompt import PromptInputs, build_system_prompt
from superclaw.provider import LitellmProvider
from superclaw.runtime import Provider
from superclaw.sandbox import Backend, detect
from superclaw.session import SessionStore, prompt_hash
from superclaw.settings import Settings
from superclaw.skills import load_skills
from superclaw.tools import Registry
from superclaw.tools.ask import AskUser
from superclaw.tools.files import core_file_tools
from superclaw.tools.plan import UpdatePlan
from superclaw.tools.search import ToolSearch
from superclaw.tools.shell import Bash
from superclaw.tools.skill import SkillTool


class NoProviderKey(RuntimeError):
    pass


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
    settings: Settings = field(default_factory=Settings.from_env)
    max_turns: int = 12
    token_budget: int = 0
    intent_gate: bool = False
    hooks: Dispatcher | None = None

    @property
    def context_window(self) -> int:
        return self.settings.context_window

    @property
    def mode(self) -> Mode:
        return self.policy.mode

    @mode.setter
    def mode(self, value: Mode) -> None:
        self.policy.mode = value

    def close(self) -> None:
        self.gs.close()


def build_hooks(settings: Settings, workspace: Path, trust_workspace: bool) -> Dispatcher | None:
    paths = [settings.user_hooks]
    if trust_workspace:
        paths.append(workspace / ".superclaw" / "hooks.json")
    hooks = load_hooks(paths)
    return Dispatcher(hooks, workspace) if hooks else None


def build_registry(memory: Memory, workspace: Path, backend: Backend | None = None, settings: Settings | None = None) -> Registry:
    roots = (settings or Settings.from_env()).skill_roots(workspace)
    registry = Registry()
    for tool in (*core_file_tools(), Bash(backend), UpdatePlan(), SkillTool(roots=roots), AskUser(),
                 memory.search_tool(), memory.note_tool()):
        registry.register(tool)
    registry.register(ToolSearch(registry))
    return registry


def build_runtime(
    settings: Settings,
    workspace: Path,
    mode: Mode,
    *,
    max_turns: int = 12,
    intent_gate: bool = False,
    hooks: Dispatcher | None = None,
) -> Runtime:
    chain = build_provider_chain([settings.model], free_first=False)
    if not chain:
        raise NoProviderKey(f"no API key resolved for model {settings.model!r}; set the provider's key (for example OPENROUTER_API_KEY)")
    settings.db_path.mkdir(parents=True, exist_ok=True)
    gs = SuperGraph(path=str(settings.db_path))
    memory = Memory(gs)
    backend = detect()
    return Runtime(
        gs=gs, store=SessionStore(gs), memory=memory, registry=build_registry(memory, workspace, backend, settings),
        policy=Policy(workspace, mode, sandboxed=backend is not None), provider=LitellmProvider(chain),
        workspace=workspace, model=settings.model, settings=settings, max_turns=max_turns,
        token_budget=settings.budget_tokens, intent_gate=intent_gate, hooks=hooks,
    )


def system_prompt_for(rt: Runtime, prompt: str) -> str:
    return build_system_prompt(PromptInputs(
        cwd=rt.workspace, mode=rt.mode, skills=load_skills(rt.settings.skill_roots(rt.workspace)),
        memory=rt.memory.recall(prompt), user_guidelines=rt.settings.user_guidelines,
        provider=rt.model.split("/", 1)[0], model=rt.model, request_kind=rt.policy.request_kind if rt.intent_gate else None,
    ))


def resolve_session(rt: Runtime, resume: str | None) -> str:
    if not resume:
        return rt.store.create(cwd=str(rt.workspace), model=rt.model)
    sid = rt.store.latest() if resume == "latest" else resume
    if not sid or rt.store.get(sid) is None:
        raise KeyError(f"no session {resume!r}")
    return sid


@dataclass
class Callbacks:
    on_event: Callable[[dict[str, Any]], None] | None = None
    on_permission: Callable[[dict[str, Any]], str] | None = None
    on_ask_user: Callable[[list[dict[str, Any]]], list[str]] | None = None


def run_once(rt: Runtime, prompt: str, sid: str, callbacks: Callbacks | None = None, *, require_completion: bool = False, verify: bool = False) -> Result:
    cb = callbacks or Callbacks()
    if rt.intent_gate:
        rt.policy.request_kind = classify(rt.provider, prompt)
        if cb.on_event:
            cb.on_event({"type": "intent", "kind": rt.policy.request_kind.value})
    system_prompt = system_prompt_for(rt, prompt)
    previous = rt.store.last_prompt(sid)
    if previous and previous.get("hash") != prompt_hash(system_prompt) and cb.on_event:
        cb.on_event({"type": "prompt_drift", "previous": previous.get("hash"), "current": prompt_hash(system_prompt)})
    return run(prompt, rt.provider, Options(
        registry=rt.registry, policy=rt.policy, workspace=rt.workspace,
        system_prompt=system_prompt, history=rt.store.replay(sid),
        max_turns=rt.max_turns, token_budget=rt.token_budget, context_window=rt.context_window,
        require_completion_signal=require_completion, verify=verify,
        on_event=cb.on_event, on_permission=cb.on_permission, on_ask_user=cb.on_ask_user,
        session=rt.store, session_id=sid, hooks=rt.hooks,
    ))
