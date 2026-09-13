from pathlib import Path

import pytest

from superclaw.agents import load_agents
from superclaw.agents import resolve as resolve_agent
from superclaw.policy import Mode
from superclaw.prompt import PromptInputs, build_system_prompt, core_prompt, project_guidelines, skills_block
from superclaw.runtime import approx_tokens
from superclaw.settings import LIMITS
from superclaw.skills import Skill, load_skills
from superclaw.usercommands import expand, load_commands
from superclaw.usercommands import find as find_command


def test_prompt_assembly_guidelines_and_skills(tmp_path):
    root = tmp_path
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    assert 0 < approx_tokens(core_prompt()) < 1000 and approx_tokens(build_system_prompt(PromptInputs(cwd=Path("/nonexistent"), mode=Mode.ASK))) < 1600
    (root / "AGENTS.md").write_text("ROOT RULES")
    (root / "SUPERCLAW.md").write_text("BRAND")
    svc = root / "services" / "api"
    svc.mkdir(parents=True)
    (svc / "agents.md").write_text("API RULES")
    out = project_guidelines(svc, root)
    assert out.index("ROOT RULES") < out.index("API RULES") and "BRAND" not in out and "## Project guidelines (services/api/agents.md)" in out
    (root / "AGENTS.md").write_text("x" * (LIMITS.guideline_file_bytes + 100))
    assert project_guidelines(root, root).count("x") <= LIMITS.guideline_file_bytes
    skill = tmp_path / "skills" / "bench"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: bench\ndescription: Run benchmarks.\n---\nBODY")
    skills = load_skills([tmp_path / "skills", tmp_path / "missing"])
    assert skills == [Skill("bench", "Run benchmarks.", "BODY", str(skill / "SKILL.md"))]
    block = skills_block([*skills, *(Skill(f"s{i:03d}", "d" * LIMITS.skill_description_chars, "", "p") for i in range(60))])
    assert "- bench: Run benchmarks." in block and "BODY" not in block and "more (call skill with a name" in block
    userfile = tmp_path / "SUPERCLAW.md"
    userfile.write_text("PERSONAL")
    prompt = build_system_prompt(PromptInputs(cwd=root, mode=Mode.PLAN, memory="- user prefers tabs", model="m", provider="p", skills=skills, user_guidelines=userfile))
    assert "Plan mode is active" in prompt and "user prefers tabs" in prompt and "Git branch: main" in prompt and prompt.index("PERSONAL") < prompt.index("xxxx")
    profiles = tmp_path / "agents"
    profiles.mkdir()
    (profiles / "reviewer.md").write_text("---\nname: reviewer\ndescription: Reviews diffs.\ntools: read_file, grep\nmodel: p/m\n---\nOnly review.")
    (profiles / "notes.txt").write_text("ignored")
    loaded = load_agents([profiles, tmp_path / "missing"])
    assert [a.name for a in loaded] == ["reviewer"] and loaded[0].tools == frozenset({"read_file", "grep"}) and loaded[0].model == "p/m"
    assert resolve_agent("reviewer", [profiles]).prompt == "Only review."
    with pytest.raises(KeyError):
        resolve_agent("nope", [profiles])
    with_agent = build_system_prompt(PromptInputs(cwd=root, mode=Mode.ASK, model="m", provider="p", agent=loaded[0].prompt))
    assert "<agent>" in with_agent and "Only review." in with_agent and "never widens" in with_agent
    assert "<agent>" not in build_system_prompt(PromptInputs(cwd=root, mode=Mode.ASK, model="m", provider="p"))
    commands = tmp_path / "commands"
    commands.mkdir()
    (commands / "pr.md").write_text("---\ndescription: Open a PR.\nagent: reviewer\n---\nOpen a PR titled $1 for $ARGUMENTS; cost $$5")
    (commands / "Bad Name.md").write_text("ignored")
    (commands / "plain.md").write_text("Just do it.")
    loaded = load_commands([commands, tmp_path / "missing"])
    assert [c.name for c in loaded] == ["plain", "pr"] and loaded[1].agent == "reviewer" and loaded[0].description == "User command: /plain"
    assert expand(loaded[1].template, "42 fix the build") == "Open a PR titled 42 for 42 fix the build; cost $5"
    assert expand("$1 then $2 then $3", "a b") == "a then b then " and expand("Just do it.", "now") == "Just do it.\n\nnow" and expand("x", "") == "x"
    assert expand("title $1", '"fix build" now') == "title fix build" and expand("$1", "it's") == "it's"
    assert find_command("PR", [commands]).name == "pr" and find_command("nope", [commands]) is None
