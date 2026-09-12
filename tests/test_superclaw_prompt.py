from pathlib import Path

from superclaw.policy import Mode
from superclaw.prompt import PromptInputs, build_system_prompt, core_prompt, find_git_root, project_guidelines, skills_block
from superclaw.runtime import approx_tokens
from superclaw.settings import LIMITS
from superclaw.skills import Skill


def repo(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    return tmp_path


def test_core_prompt_and_confirmation_policy_stay_small():
    assert 0 < approx_tokens(core_prompt()) < 1000
    prompt = build_system_prompt(PromptInputs(cwd=Path("/nonexistent"), mode=Mode.ASK))
    assert "## Confirmation policy" in prompt and approx_tokens(prompt) < 1600


def test_project_guidelines_walk_root_to_cwd_with_priority_and_caps(tmp_path):
    root = repo(tmp_path)
    sub = root / "a" / "b"
    sub.mkdir(parents=True)
    assert find_git_root(sub) == root and find_git_root(tmp_path.parent) is None
    (root / "AGENTS.md").write_text("ROOT RULES")
    (root / "SUPERCLAW.md").write_text("BRAND")
    (root / ".superclaw").mkdir()
    (root / ".superclaw" / "AGENTS.md").write_text("HIDDEN")
    svc = root / "services" / "api"
    svc.mkdir(parents=True)
    (svc / "agents.md").write_text("API RULES")
    out = project_guidelines(svc, root)
    assert out.index("ROOT RULES") < out.index("API RULES") and "## Project guidelines (services/api/agents.md)" in out
    assert "BRAND" not in out and "HIDDEN" not in out
    (root / "AGENTS.md").unlink()
    (root / "SUPERCLAW.md").unlink()
    assert "HIDDEN" in project_guidelines(root, root)
    (root / "AGENTS.md").write_text("x" * (LIMITS.guideline_file_bytes + 100))
    out = project_guidelines(root, root)
    assert "… (truncated)" in out and out.count("x") <= LIMITS.guideline_file_bytes
    letters = "bfhkqv"
    cur = root
    for i in range(6):
        (cur / "AGENTS.md").write_text(letters[i] * LIMITS.guideline_file_bytes)
        cur = cur / f"d{i}"
        cur.mkdir()
    assert sum(project_guidelines(cur, root).count(letters[i]) for i in range(6)) <= LIMITS.guideline_total_bytes
    (tmp_path / "solo").mkdir()
    (tmp_path / "solo" / "AGENTS.md").write_text("ONLY")
    assert "ONLY" in project_guidelines(tmp_path / "solo", None)


def test_skills_block_lists_names_only_and_summarises_overflow():
    assert skills_block([]) == ""
    block = skills_block([Skill("bench", "Run benchmarks.", "SECRET BODY", "p")])
    assert "- bench: Run benchmarks." in block and "SECRET BODY" not in block and block.startswith("<available_skills>")
    block = skills_block([Skill(f"skill-{i:03d}", "d" * 200, "", "p") for i in range(60)])
    assert "more (call skill with a name" in block and len(block) < LIMITS.skills_index_bytes + 300


def test_assembly_orders_user_guidelines_before_project_and_adds_mode_memory_environment(tmp_path):
    root = repo(tmp_path)
    (root / "AGENTS.md").write_text("PROJECT")
    userfile = tmp_path / "SUPERCLAW.md"
    userfile.write_text("PERSONAL")
    prompt = build_system_prompt(PromptInputs(cwd=root, mode=Mode.PLAN, memory="- user prefers tabs", model="m", provider="p",
                                              skills=[Skill("s", "d", "", "p")], user_guidelines=userfile))
    assert "Plan mode is active" in prompt and "<memory>" in prompt and "user prefers tabs" in prompt
    assert f"Working directory: {root}" in prompt and "Git branch: main" in prompt and "Active model: m" in prompt and "<available_skills>" in prompt
    assert "## User guidelines" in prompt and prompt.index("PERSONAL") < prompt.index("PROJECT")
