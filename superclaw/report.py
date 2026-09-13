from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from superclaw.app import Runtime
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
                          user_guidelines=rt.settings.user_guidelines, provider=rt.model.split("/", 1)[0], model=rt.model)
    eager = rt.registry.definitions(rt.policy.visible, set())
    latest = rt.store.latest()
    categories = {
        "system prompt": approx_tokens(core_prompt()) + approx_tokens(confirmation_policy()) + approx_tokens(environment_block(inputs.cwd)),
        "user guidelines": approx_tokens(user_guidelines(inputs.user_guidelines)),
        "project guidelines": approx_tokens(project_guidelines(inputs.cwd, find_git_root(inputs.cwd))),
        "skills index": approx_tokens(skills_block(skills)),
        "memory recall": approx_tokens(inputs.memory),
        "tool schemas": sum(approx_tokens(json.dumps(t)) + LIMITS.message_overhead_tokens for t in eager),
        "history": sum(approx_tokens(m.content) for m in rt.store.replay(latest)) if latest else 0,
    }
    return ContextReport(window=rt.context_window, categories=categories)


def doctor_lines(rt: Runtime, setup_hint: str) -> list[str]:
    dot, env, glyphs = rt.settings.glyphs.dot, os.environ, rt.settings.glyphs
    return [
        f"terminal {env.get('TERM_PROGRAM') or env.get('TERM') or 'unknown'} {dot} vte {env.get('VTE_VERSION') or 'n/a'} "
        f"{dot} glyphs {'ascii' if glyphs.border == ASCII.border else 'unicode'}",
        f"sandbox {'on' if rt.policy.sandboxed else 'off'} {dot} mode {rt.mode.value} {dot} effort {rt.settings.effort or EFFORT_OFF}",
        f"model {rt.model} {dot} window {compact(rt.context_window)} {dot} {'catalog' if rt.model_info.known else 'fallback'}",
        f"store {rt.settings.db_path} {dot} workspace {rt.workspace}",
        "providers with a key: " + (f" {dot} ".join(p.name for p in keyed_providers()) or f"none; {setup_hint}"),
    ]
