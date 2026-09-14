from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from superclaw.app import Runtime, repo_map_text
from superclaw.catalog import keyed_providers
from superclaw.prompt import (
    PromptInputs,
    confirmation_policy,
    core_prompt,
    environment_block,
    find_git_root,
    project_guidelines,
    skills_block,
    user_guidelines,
)
from superclaw.runtime import approx_tokens, compact
from superclaw.settings import ASCII, EFFORT_OFF, LIMITS
from superclaw.skills import load_skills
from superclaw.maintain import health_line
from superclaw.tooling import host_tools

_PERCENT = 100


@dataclass
class ContextReport:
    window: int
    categories: dict[str, int] = field(default_factory=dict)

    @property
    def used(self) -> int:
        return sum(self.categories.values())

    @property
    def free(self) -> int:
        return max(0, self.window - self.used)

    def percent(self, tokens: int) -> float:
        return _PERCENT * tokens / self.window if self.window else 0.0


def context_report(rt: Runtime, prompt: str = "") -> ContextReport:
    skills = load_skills(rt.settings.skill_roots(rt.workspace))
    inputs = PromptInputs(cwd=rt.workspace, mode=rt.mode, skills=skills, memory=rt.memory.recall(prompt) if prompt else "",
                          user_guidelines=rt.settings.user_guidelines, extra_dirs=rt.extra_dirs,
                          provider=rt.model.split("/", 1)[0], model=rt.model, claude_config=rt.settings.claude_config, sandbox=rt.sandbox)
    eager = rt.registry.definitions(rt.policy.visible, set())
    latest = rt.store.latest()
    categories = {
        "system prompt": approx_tokens(core_prompt()) + approx_tokens(confirmation_policy()) + approx_tokens(environment_block(inputs.cwd, inputs.extra_dirs, inputs.tools, inputs.sandbox)),
        "user guidelines": approx_tokens(user_guidelines(inputs.user_guidelines)),
        "project guidelines": approx_tokens(project_guidelines(inputs.cwd, find_git_root(inputs.cwd), inputs.claude_config)),
        "skills index": approx_tokens(skills_block(skills)),
        "repo map": approx_tokens(repo_map_text(rt)),
        "memory recall": approx_tokens(inputs.memory),
        "tool schemas": sum(approx_tokens(json.dumps(t)) + LIMITS.message_overhead_tokens for t in eager),
        "history": sum(approx_tokens(m.content) for m in rt.store.replay(latest)) if latest else 0,
    }
    return ContextReport(window=rt.context_window, categories=categories)


def graph_line(gs: Any, dot: str) -> str:
    if gs is None:
        return "graph not opened by this command; /doctor inside the TUI shows the embedder and the counts"
    embedders = [e for e in (gs.execute("SYS EMBEDDERS").data or []) if e.get("status") == "active"]
    stats = gs.execute("SYS STATS").data or {}
    embedder = f"{embedders[0]['name']} {embedders[0].get('dims', '?')}d" if embedders else "no embedder, lexical recall only"
    return f"graph {embedder} {dot} {int(stats.get('node_count', 0)):,} nodes {dot} {int(stats.get('edge_count', 0)):,} edges"


def cache_line(rt: Runtime, dot: str) -> str:
    if rt.store is None:
        return "prompt cache not measured by this command; /doctor inside the TUI reads the sessions"
    sent = cached = 0
    for session in rt.store.recent():
        if session["model"] != rt.model:
            continue
        use = rt.store.usage(session["id"])
        sent += use["tokens"]
        cached += use["cached"]
    if not sent:
        return f"prompt cache no billed turns yet for {rt.model}"
    share = cached / sent
    verdict = "this provider is not serving the repeated prefix from cache; a turn costs full price" if share < LIMITS.cache_hit_floor else "the repeated prefix is being served from cache"
    return f"prompt cache {share:.0%} of {sent:,} tokens across recent sessions {dot} {verdict}"


def doctor_lines(rt: Runtime, setup_hint: str) -> list[str]:
    dot, env, glyphs = rt.settings.glyphs.dot, os.environ, rt.settings.glyphs
    return [
        f"terminal {env.get('TERM_PROGRAM') or env.get('TERM') or 'unknown'} {dot} vte {env.get('VTE_VERSION') or 'n/a'} "
        f"{dot} glyphs {'ascii' if glyphs.border == ASCII.border else 'unicode'}",
        f"sandbox {'on' if rt.policy.sandboxed else 'off'} {dot} mode {rt.mode.value} {dot} effort {rt.settings.effort or EFFORT_OFF}"
        + (f" {dot} agent {rt.agent.name}" if rt.agent else ""),
        f"model {rt.model} {dot} window {compact(rt.context_window)} {dot} {'catalog' if rt.model_info.known else 'fallback'}"
        + (f" {dot} then {', '.join(rt.settings.fallback_models)}" if rt.settings.fallback_models else ""),
        f"store {rt.settings.db_path}{f' ({rt.gs.role})' if hasattr(rt.gs, 'role') else ''} {dot} workspace {rt.workspace}",
        f"mcp {len(rt.mcp.tools) if rt.mcp else 0} tools {dot} {len(rt.mcp.clients) if rt.mcp else 0} servers"
        + (f" {dot} {len(rt.mcp.skipped)} skipped" if rt.mcp and rt.mcp.skipped else ""),
        graph_line(rt.gs, dot),
        health_line(rt.gs, dot),
        "host tools " + (" ".join(host_tools()) or "none of the modern set"),
        f"claude config {'on' if rt.settings.claude_config else 'off'} {dot} {rt.settings.claude_dir}",
        cache_line(rt, dot),
        "providers with a key: " + (f" {dot} ".join(p.name for p in keyed_providers()) or f"none; {setup_hint}"),
    ]
