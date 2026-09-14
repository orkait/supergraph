from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from superclaw.agents import Agent, load_agents
from superclaw.catalog import provider_of
from superclaw.delegate import Delegate
from superclaw.documents import Documents
from superclaw.hooks import Dispatcher, load_hooks
from superclaw.intent import classify
from superclaw.kernel import READ_VERBS, Kernel, Python
from superclaw import maintain
from superclaw.loop import Options, Result, run
from superclaw.mcp import Bridge, Source, claude_sources, connect_all, load_config
from superclaw.memory import Memory
from superclaw.models import ModelInfo
from superclaw.observations import ObservationStore, Recall
from superclaw.policy import Mode, Policy
from superclaw.prompt import PromptInputs, build_system_prompt
from superclaw.prompt import _git_branch
from superclaw.provider import LitellmProvider
from superclaw.repomap import render, scan
from superclaw.runtime import Message, Provider, approx_tokens
from superclaw.sandbox import Backend, detect
from superclaw.session import SessionStore, prompt_hash
from superclaw.share import open_shared
from superclaw.settings import LIMITS, MCP_FILE, PROVIDERS, SESSION_END_OTHER, UNSAFE_SNAPSHOT, WORKSPACE_DIR, Settings
from superclaw.skills import load_skills
from superclaw.tooling import host_tools
from superclaw.tools import Registry
from superclaw.tools.ask import AskUser
from superclaw.tools.download import Download
from superclaw.tools.fetch import WebFetch
from superclaw.tools.files import core_file_tools
from superclaw.tools.ingest import Ingest
from superclaw.tools.plan import UpdatePlan
from superclaw.tools.search import ToolSearch
from superclaw.tools.shell import Bash, BashOutput, Jobs
from superclaw.tools.skill import SkillTool
from superclaw.tools.web import WebSearch


class NoProviderKey(RuntimeError):
    pass


_started: set[str] = set()


@dataclass
class Runtime:
    gs: Any
    store: SessionStore | None
    memory: Memory | None
    registry: Registry
    policy: Policy
    provider: Provider | None
    workspace: Path
    model: str
    settings: Settings = field(default_factory=Settings.from_env)
    extra_dirs: tuple[Path, ...] = ()
    max_turns: int = LIMITS.max_turns
    token_budget: int = 0
    intent_gate: bool = False
    hooks: Dispatcher | None = None
    kernel: Kernel | None = None
    mcp: Bridge | None = None
    agent: Agent | None = None
    session_id: str = ""
    sandbox: str = ""

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

    def close(self, reason: str = SESSION_END_OTHER) -> None:
        if (bash := self.registry.get("bash")) is not None and hasattr(bash, "jobs"):
            bash.jobs.close()
        if self.hooks:
            self.hooks.dispatch("sessionEnd", {"session": self.session_id, "reason": reason}, reason)
        if self.mcp:
            self.mcp.close()
        if self.kernel:
            self.kernel.close()
        if self.gs is not None:
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


def build_kernel(workspace: Path, backend: Backend | None, observations: ObservationStore, gs: Any,
                 extra_dirs: tuple[Path, ...] = ()) -> Kernel:
    return Kernel(workspace, backend, kernel_resolver(observations, gs), extra_dirs)


def build_hooks(settings: Settings, workspace: Path, trust_workspace: bool) -> Dispatcher | None:
    entries: list[Path | tuple[Path, Path | None]] = [settings.user_hooks]
    if trust_workspace:
        entries.append(workspace / WORKSPACE_DIR / "hooks.json")
    entries += settings.claude_settings(workspace, trust_workspace)
    entries += [(plugin.hooks, plugin.path) for plugin in settings.plugins(workspace, trusted=trust_workspace) if plugin.hooks]
    hooks = load_hooks(entries)
    return Dispatcher(hooks, workspace) if hooks else None


def mcp_paths(settings: Settings, workspace: Path, trust_workspace: bool) -> list[Path | Source]:
    paths: list[Path | Source] = [settings.user_mcp]
    if trust_workspace:
        paths.append(workspace / WORKSPACE_DIR / MCP_FILE)
    if settings.claude_config:
        paths += claude_sources(settings.claude_state, workspace, trust_workspace)
    return paths + [plugin.mcp for plugin in settings.plugins(workspace, trusted=trust_workspace) if plugin.mcp]


def build_registry(memory: Memory, observations: ObservationStore, workspace: Path, backend: Backend | None = None,
                   settings: Settings | None = None, kernel: Kernel | None = None, sessions: SessionStore | None = None,
                   documents: Documents | None = None) -> Registry:
    settings = settings or Settings.from_env()
    registry = Registry(observations=observations)
    roots = settings.skill_roots(workspace)
    jobs = Jobs()
    for tool in (*core_file_tools(), Bash(backend, kernel, jobs), BashOutput(jobs), UpdatePlan(), SkillTool(roots=roots), AskUser(), WebSearch(settings), WebFetch(observations, settings, memory.facts),
                 Download(documents), Ingest(documents), memory.search_tool(), memory.note_tool(), Recall(observations, sessions, memory.facts, documents), Delegate(),
                 *([Python(kernel)] if kernel else [])):
        registry.register(tool)
    registry.register(ToolSearch(registry))
    return registry


def build_runtime(
    settings: Settings,
    workspace: Path,
    mode: Mode,
    *,
    max_turns: int = LIMITS.max_turns,
    intent_gate: bool = False,
    hooks: Dispatcher | None = None,
    require_provider: bool = True,
    allow_tools: frozenset[str] = frozenset(),
    deny_tools: frozenset[str] = frozenset(),
    extra_dirs: tuple[Path, ...] = (),
    mcp_config: list[Path | Source] | None = None,
    agent: Agent | None = None,
    open_store: bool = True,
) -> Runtime:
    provider = connect_provider(settings.model, settings.effort, settings.fallback_models, settings.stream, settings.output_cap())
    if provider is None and require_provider:
        raise NoProviderKey(f"no API key resolved for model {settings.model!r}; run `superclaw setup` or set the provider's key (for example OPENROUTER_API_KEY)")
    backend = detect()
    gs = memory = kernel = sessions = None
    registry = Registry()
    if open_store:
        settings.db_path.mkdir(parents=True, exist_ok=True)
        gs = open_shared(settings.db_path, reader_for(provider))
        memory = Memory(gs)
        observations = ObservationStore(gs)
        sessions = SessionStore(gs)
        kernel = build_kernel(workspace, backend, observations, gs, extra_dirs)
        registry = build_registry(memory, observations, workspace, backend, settings, kernel, sessions, Documents(gs))
        if maintain.stale(gs):
            maintain.maintain(gs, optimize=False)
    bridge = connect_all(load_config(mcp_config), registry) if mcp_config else None
    policy = Policy(workspace, mode, sandboxed=backend is not None, allow_tools=allow_tools, deny_tools=deny_tools, extra_dirs=extra_dirs)
    if agent:
        policy.scope_to(agent.tools)
    return Runtime(
        gs=gs, store=sessions, memory=memory, registry=registry, policy=policy,
        provider=provider, workspace=workspace, model=settings.model, settings=settings, extra_dirs=extra_dirs, max_turns=max_turns,
        token_budget=settings.budget_tokens, intent_gate=intent_gate, hooks=hooks, kernel=kernel, mcp=bridge, agent=agent,
        sandbox=getattr(backend, "name", "") if backend else "",
    )


def reader_for(provider: Provider | None) -> Callable[[str], str] | None:
    if provider is None:
        return None
    return lambda prompt: provider.complete([Message(role="user", content=prompt)], []).text


def connect_provider(model: str, effort: str = "", fallbacks: tuple[str, ...] = (), stream: bool = True, max_tokens: int = LIMITS.completion_max_tokens) -> Provider | None:
    def resolve() -> list[dict[str, Any]]:
        from supergraph.ingest.llm.resolve import build_provider_chain

        return build_provider_chain([model, *fallbacks], free_first=False)

    known = provider_of(model)
    if known is not None and not os.environ.get(known.env):
        return None
    if known is None and not resolve():
        return None
    return LitellmProvider(resolve if known is not None else resolve(), effort=effort, stream=stream, max_tokens=max_tokens)


def switch_model(rt: Runtime, model: str) -> None:
    provider = provider_of(model)
    if provider is None:
        raise KeyError(f"unknown provider in {model!r}; providers: {', '.join(p.name for p in PROVIDERS)}")
    if not os.environ.get(provider.env):
        raise NoProviderKey(f"no {provider.env} for {provider.name}")
    rt.settings.save_model(model)
    rt.settings = replace(rt.settings, model=model)
    rt.model = model
    rt.provider = connect_provider(model, rt.settings.effort, rt.settings.fallback_models, rt.settings.stream, rt.settings.output_cap())


def apply_effort(rt: Runtime, effort: str) -> None:
    rt.settings.save_effort(effort)
    rt.settings = replace(rt.settings, effort=effort)
    if rt.provider is not None:
        rt.provider.effort = effort


def repo_map_text(rt: Runtime) -> str:
    return render(scan(rt.workspace)) if rt.settings.repo_map else ""


@dataclass
class Context:
    system_prompt: str
    memories: int
    facts: int
    skills: int
    repo_files: int

    def event(self, history: int) -> dict[str, Any]:
        return {"type": "context", "memories": self.memories, "facts": self.facts, "skills": self.skills, "repo_files": self.repo_files,
                "history": history, "prompt_tokens": approx_tokens(self.system_prompt)}


def context_for(rt: Runtime, prompt: str) -> Context:
    hits = rt.memory.hits(prompt)
    facts = rt.memory.facts.search(prompt)
    skills = load_skills(rt.settings.skill_roots(rt.workspace))
    found = scan(rt.workspace) if rt.settings.repo_map else None
    system_prompt = build_system_prompt(PromptInputs(
        cwd=rt.workspace, mode=rt.mode, skills=skills,
        memory=rt.memory.render(hits), facts=rt.memory.facts.render(facts), user_guidelines=rt.settings.user_guidelines, extra_dirs=rt.extra_dirs,
        agent=rt.agent.prompt if rt.agent else "", repo_map=render(found) if found else "",
        provider=rt.model.split("/", 1)[0], model=rt.model, request_kind=rt.policy.request_kind if rt.intent_gate else None,
        tools=host_tools(), claude_config=rt.settings.claude_config, sandbox=rt.sandbox,
    ))
    return Context(system_prompt, len(hits), len(facts), len(skills), len(found.files) if found else 0)


def system_prompt_for(rt: Runtime, prompt: str) -> str:
    return context_for(rt, prompt).system_prompt


def resolve_session(rt: Runtime, resume: str | None, fork: str | None = None) -> str:
    if fork:
        source = rt.store.latest() if fork == "latest" else fork
        if not source or rt.store.get(source) is None:
            raise KeyError(f"no session {fork!r}")
        rt.session_id = rt.store.fork(source)
    elif not resume:
        rt.session_id = rt.store.create(cwd=str(rt.workspace), model=rt.model, branch=_git_branch(rt.workspace))
    else:
        sid = rt.store.latest() if resume == "latest" else resume
        if not sid or rt.store.get(sid) is None:
            raise KeyError(f"no session {resume!r}")
        rt.session_id = sid
    return rt.session_id


@dataclass
class Callbacks:
    on_event: Callable[[dict[str, Any]], None] | None = None
    on_permission: Callable[[dict[str, Any]], str] | None = None
    on_ask_user: Callable[[list[dict[str, Any]]], list[str]] | None = None


def run_once(rt: Runtime, prompt: str, sid: str, callbacks: Callbacks | None = None, *, require_completion: bool = False, verify: bool = False,
             cancelled: Callable[[], bool] | None = None, images: list[str] | None = None) -> Result:
    cb = callbacks or Callbacks()
    if rt.provider is None:
        raise NoProviderKey("no provider connected; run setup first")
    if rt.mode is Mode.UNSAFE and rt.gs is not None and maintain.snapshot(rt.gs, f"{UNSAFE_SNAPSHOT}-{sid}") and cb.on_event:
        cb.on_event({"type": "snapshot", "name": f"{UNSAFE_SNAPSHOT}-{sid}"})
    if rt.intent_gate:
        rt.policy.request_kind = classify(rt.provider, prompt)
        if cb.on_event:
            cb.on_event({"type": "intent", "kind": rt.policy.request_kind.value})
    rt.store.name_once(sid, prompt)
    context = context_for(rt, prompt)
    system_prompt = context.system_prompt
    history = rt.store.replay(sid)
    if cb.on_event:
        cb.on_event(context.event(len(history)))
    if rt.kernel and not rt.kernel.alive and rt.registry.observations:
        rt.kernel.restore(rt.registry.observations.load_kernel(sid))
    previous = rt.store.last_prompt(sid)
    if previous and previous.get("hash") != prompt_hash(system_prompt) and cb.on_event:
        cb.on_event({"type": "prompt_drift", "previous": previous.get("hash"), "current": prompt_hash(system_prompt)})
    result = run(prompt, rt.provider, Options(
        registry=rt.registry, policy=rt.policy, workspace=rt.workspace, extra_dirs=rt.extra_dirs, images=images or [],
        system_prompt=system_prompt, history=history,
        max_turns=rt.max_turns, token_budget=rt.token_budget, budget_usd=rt.settings.budget_usd,
        context_window=rt.context_window, model_info=rt.model_info,
        agents={a.name: a for a in load_agents(rt.settings.agent_roots(rt.workspace))},
        require_completion_signal=require_completion, verify=verify,
        on_event=cb.on_event, on_permission=cb.on_permission, on_ask_user=cb.on_ask_user,
        session=rt.store, session_id=sid, hooks=rt.hooks, cancelled=cancelled, session_start=sid not in _started,
    ))
    _started.add(sid)
    if rt.kernel and rt.registry.observations and (blob := rt.kernel.checkpoint()):
        rt.registry.observations.save_kernel(sid, blob)
    return result
