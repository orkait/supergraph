from __future__ import annotations

import os
import shlex
from typing import override

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.agents.model_connection import ModelConnectionSpec
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

SOURCE = "supergraphdb[superclaw] @ git+https://github.com/orkait/supergraph"
BRAIN = "/tmp/superclaw-brain"
LOG = "/logs/agent/superclaw.jsonl"
PROVIDER_KEYS = (
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "XAI_API_KEY",
)


class Superclaw(BaseInstalledAgent):
    """Installs superclaw into the task container and runs it headless."""

    MODEL_CONNECTION = ModelConnectionSpec(passthrough=True)

    @staticmethod
    @override
    def name() -> str:
        return "superclaw"

    @override
    def get_version_command(self) -> str | None:
        return ". $HOME/.local/bin/env 2>/dev/null; superclaw --version 2>&1 | head -1"

    @override
    def parse_version(self, stdout: str) -> str:
        return stdout.strip().splitlines()[0] if stdout.strip() else ""

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(environment, ("curl", "git"))
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi; "
                'if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; fi; '
                f'uv tool install --python 3.13 "{SOURCE}" && '
                "superclaw --help >/dev/null"
            ),
        )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        pass

    @override
    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        access = self.model_connection
        env = {key: os.environ[key] for key in PROVIDER_KEYS if os.environ.get(key)}
        env.update(access.env)
        if self.model_name:
            env["SUPERCLAW_MODEL"] = self.model_name
        await self.exec_as_agent(
            environment,
            command=(
                '. $HOME/.local/bin/env 2>/dev/null; mkdir -p /logs/agent; '
                f"superclaw --dangerously-skip-permissions --db {BRAIN} "
                f"exec --output-format stream-json {shlex.quote(instruction)} "
                f"2>&1 | stdbuf -oL tee {LOG}"
            ),
            env=env,
        )
