from __future__ import annotations

import json
from dataclasses import dataclass, field

from superclaw.app import Runtime
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
from superclaw.runtime import approx_tokens
from superclaw.settings import LIMITS
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
