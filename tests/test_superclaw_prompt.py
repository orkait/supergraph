from pathlib import Path

from superclaw.policy import Mode
from superclaw.prompt import PromptInputs, build_system_prompt, core_prompt, project_guidelines, skills_block
from superclaw.runtime import approx_tokens
from superclaw.settings import LIMITS
from superclaw.skills import Skill, load_skills


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
