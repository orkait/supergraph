from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from supergraph import SuperGraph
from supergraph.ingest.llm.resolve import build_provider_chain

from superclaw.catalog import provider_of
from superclaw.delegate import Delegate
from superclaw.hooks import Dispatcher, load_hooks
from superclaw.intent import classify
from superclaw.kernel import READ_VERBS, Kernel, Python
from superclaw.loop import Options, Result, run
from superclaw.memory import Memory
from superclaw.models import ModelInfo
from superclaw.observations import ObservationStore, Recall
from superclaw.policy import Mode, Policy
from superclaw.prompt import PromptInputs, build_system_prompt
from superclaw.provider import LitellmProvider
from superclaw.runtime import Provider
from superclaw.sandbox import Backend, detect
from superclaw.session import SessionStore, prompt_hash
from superclaw.settings import PROVIDERS, Settings
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
    provider: Provider | None
    workspace: Path
    model: str
    settings: Settings = field(default_factory=Settings.from_env)
    max_turns: int = 12
    token_budget: int = 0
    intent_gate: bool = False
    hooks: Dispatcher | None = None
    kernel: Kernel | None = None

    @property
    def model_info(self) -> ModelInfo:
        return self.settings.model_info()

    @property
    def context_window(self) -> int:
        return self.settings.window()

    @property
    def mode(self) -> Mode:
        return self.policy.mode

    @mode.setter
    def mode(self, value: Mode) -> None:
        self.policy.mode = value

    def close(self) -> None:
        if self.kernel:
            self.kernel.close()
        self.gs.close()


def kernel_resolver(observations: ObservationStore, gs: Any) -> Callable[[str, dict[str, Any]], Any]:
    def resolve(kind: str, request: dict[str, Any]) -> Any:
        if kind == "obs":
            found = observations.load(str(request.get("ref") or ""))
            if found is None:
                raise KeyError(f"no stored result §{request.get('ref')}")
            return found.body
        if kind == "query":
            dsl = str(request.get("dsl") or "").strip()
            if not dsl.upper().startswith(READ_VERBS):
                raise PermissionError("only read queries are allowed from the kernel")
            return gs.execute(dsl).data
        raise ValueError(f"unknown kernel request {kind!r}")

    return resolve


def build_kernel(workspace: Path, backend: Backend | None, observations: ObservationStore, gs: Any) -> Kernel:
    return Kernel(workspace, backend, kernel_resolver(observations, gs))


def build_hooks(settings: Settings, workspace: Path, trust_workspace: bool) -> Dispatcher | None:
    paths = [settings.user_hooks]
    if trust_workspace:
        paths.append(workspace / ".superclaw" / "hooks.json")
    hooks = load_hooks(paths)
    return Dispatcher(hooks, workspace) if hooks else None


def build_registry(memory: Memory, observations: ObservationStore, workspace: Path, backend: Backend | None = None,
                   settings: Settings | None = None, kernel: Kernel | None = None) -> Registry:
    settings = settings or Settings.from_env()
    registry = Registry(observations=observations)
    roots = settings.skill_roots(workspace)
    for tool in (*core_file_tools(), Bash(backend, kernel), UpdatePlan(), SkillTool(roots=roots), AskUser(),
                 memory.search_tool(), memory.note_tool(), Recall(observations), Delegate(), *([Python(kernel)] if kernel else [])):
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
    require_provider: bool = True,
) -> Runtime:
    provider = connect_provider(settings.model, settings.effort)
    if provider is None and require_provider:
        raise NoProviderKey(f"no API key resolved for model {settings.model!r}; run `superclaw setup` or set the provider's key (for example OPENROUTER_API_KEY)")
    settings.db_path.mkdir(parents=True, exist_ok=True)
    gs = SuperGraph(path=str(settings.db_path))
    memory = Memory(gs)
    backend = detect()
    observations = ObservationStore(gs)
    kernel = build_kernel(workspace, backend, observations, gs)
    return Runtime(
        gs=gs, store=SessionStore(gs), memory=memory, registry=build_registry(memory, observations, workspace, backend, settings, kernel),
        policy=Policy(workspace, mode, sandboxed=backend is not None), provider=provider,
        workspace=workspace, model=settings.model, settings=settings, max_turns=max_turns,
        token_budget=settings.budget_tokens, intent_gate=intent_gate, hooks=hooks, kernel=kernel,
    )


def connect_provider(model: str, effort: str = "") -> Provider | None:
    chain = build_provider_chain([model], free_first=False)
    return LitellmProvider(chain, effort=effort) if chain else None


def switch_model(rt: Runtime, model: str) -> None:
    provider = provider_of(model)
    if provider is None:
        raise KeyError(f"unknown provider in {model!r}; providers: {', '.join(p.name for p in PROVIDERS)}")
    if not os.environ.get(provider.env):
        raise NoProviderKey(f"no {provider.env} for {provider.name}")
    rt.settings.save_model(model)
    rt.settings = replace(rt.settings, model=model)
    rt.model = model
    rt.provider = connect_provider(model, rt.settings.effort)


def apply_effort(rt: Runtime, effort: str) -> None:
    rt.settings.save_effort(effort)
    rt.settings = replace(rt.settings, effort=effort)
    if rt.provider is not None:
        rt.provider.effort = effort


def system_prompt_for(rt: Runtime, prompt: str) -> str:
    return build_system_prompt(PromptInputs(
        cwd=rt.workspace, mode=rt.mode, skills=load_skills(rt.settings.skill_roots(rt.workspace)),
        memory=rt.memory.recall(prompt), user_guidelines=rt.settings.user_guidelines,
        provider=rt.model.split("/", 1)[0], model=rt.model, request_kind=rt.policy.request_kind if rt.intent_gate else None,
    ))


def resolve_session(rt: Runtime, resume: str | None, fork: str | None = None) -> str:
    if fork:
        source = rt.store.latest() if fork == "latest" else fork
        if not source or rt.store.get(source) is None:
            raise KeyError(f"no session {fork!r}")
        return rt.store.fork(source)
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


def run_once(rt: Runtime, prompt: str, sid: str, callbacks: Callbacks | None = None, *, require_completion: bool = False, verify: bool = False,
             cancelled: Callable[[], bool] | None = None) -> Result:
    cb = callbacks or Callbacks()
    if rt.provider is None:
        raise NoProviderKey("no provider connected; run setup first")
    if rt.intent_gate:
        rt.policy.request_kind = classify(rt.provider, prompt)
        if cb.on_event:
            cb.on_event({"type": "intent", "kind": rt.policy.request_kind.value})
    system_prompt = system_prompt_for(rt, prompt)
    if rt.kernel and not rt.kernel.alive and rt.registry.observations:
        rt.kernel.restore(rt.registry.observations.load_kernel(sid))
    previous = rt.store.last_prompt(sid)
    if previous and previous.get("hash") != prompt_hash(system_prompt) and cb.on_event:
        cb.on_event({"type": "prompt_drift", "previous": previous.get("hash"), "current": prompt_hash(system_prompt)})
    result = run(prompt, rt.provider, Options(
        registry=rt.registry, policy=rt.policy, workspace=rt.workspace,
        system_prompt=system_prompt, history=rt.store.replay(sid),
        max_turns=rt.max_turns, token_budget=rt.token_budget, budget_usd=rt.settings.budget_usd,
        context_window=rt.context_window, model_info=rt.model_info,
        require_completion_signal=require_completion, verify=verify,
        on_event=cb.on_event, on_permission=cb.on_permission, on_ask_user=cb.on_ask_user,
        session=rt.store, session_id=sid, hooks=rt.hooks, cancelled=cancelled,
    ))
    if rt.kernel and rt.registry.observations and (blob := rt.kernel.checkpoint()):
        rt.registry.observations.save_kernel(sid, blob)
    return result
