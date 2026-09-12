from pathlib import Path

from superclaw.policy import Mode
from superclaw.prompt import (
    MAX_FILE_BYTES,
    MAX_TOTAL_BYTES,
    PromptInputs,
    build_system_prompt,
    core_prompt,
    find_git_root,
    project_guidelines,
    skills_block,
)
from superclaw.runtime import approx_tokens
from superclaw.skills import Skill


def repo(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    return tmp_path


class TestCorePrompt:
    def test_core_prompt_is_under_a_thousand_tokens(self):
        assert 0 < approx_tokens(core_prompt()) < 1000

    def test_confirmation_policy_is_included_and_bounded(self):
        prompt = build_system_prompt(PromptInputs(cwd=Path("/nonexistent"), mode=Mode.ASK))
        assert "## Confirmation policy" in prompt
        assert approx_tokens(prompt) < 1600


class TestGitRoot:
    def test_finds_nearest_ancestor_with_git(self, tmp_path):
        root = repo(tmp_path)
        sub = root / "a" / "b"
        sub.mkdir(parents=True)
        assert find_git_root(sub) == root

    def test_none_without_git(self, tmp_path):
        assert find_git_root(tmp_path) is None


class TestProjectGuidelines:
    def test_walks_root_to_cwd_general_to_specific(self, tmp_path):
        root = repo(tmp_path)
        (root / "AGENTS.md").write_text("ROOT RULES")
        svc = root / "services" / "api"
        svc.mkdir(parents=True)
        (svc / "agents.md").write_text("API RULES")
        out = project_guidelines(svc, root)
        assert out.index("ROOT RULES") < out.index("API RULES")
        assert "## Project guidelines (AGENTS.md)" in out
        assert "## Project guidelines (services/api/agents.md)" in out

    def test_priority_order_at_one_level(self, tmp_path):
        root = repo(tmp_path)
        (root / "SUPERCLAW.md").write_text("BRAND")
        (root / "AGENTS.md").write_text("CLASSIC")
        assert "CLASSIC" in project_guidelines(root, root)
        assert "BRAND" not in project_guidelines(root, root)

    def test_hidden_project_file_is_last_resort(self, tmp_path):
        root = repo(tmp_path)
        (root / ".superclaw").mkdir()
        (root / ".superclaw" / "AGENTS.md").write_text("HIDDEN")
        assert "HIDDEN" in project_guidelines(root, root)

    def test_per_file_cap(self, tmp_path):
        root = repo(tmp_path)
        (root / "AGENTS.md").write_text("x" * (MAX_FILE_BYTES + 100))
        out = project_guidelines(root, root)
        assert "… (truncated)" in out
        assert out.count("x") <= MAX_FILE_BYTES

    def test_total_cap_across_files(self, tmp_path):
        root = repo(tmp_path)
        # Letters chosen to not appear in headers/labels ("## Project guidelines (dN/.../AGENTS.md)").
        letters = "bfhkqv"
        cur = root
        for i in range(6):
            (cur / "AGENTS.md").write_text(letters[i] * MAX_FILE_BYTES)
            cur = cur / f"d{i}"
            cur.mkdir()
        out = project_guidelines(cur, root)
        assert sum(out.count(letters[i]) for i in range(6)) <= MAX_TOTAL_BYTES

    def test_no_git_root_uses_cwd_only(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("ONLY")
        assert "ONLY" in project_guidelines(tmp_path, None)


class TestSkillsBlock:
    def test_empty_is_empty(self):
        assert skills_block([]) == ""

    def test_lists_name_and_description_only(self):
        block = skills_block([Skill("bench", "Run benchmarks.", "SECRET BODY", "p")])
        assert "- bench: Run benchmarks." in block
        assert "SECRET BODY" not in block
        assert block.startswith("<available_skills>")

    def test_budget_summarizes_overflow_as_count(self):
        skills = [Skill(f"skill-{i:03d}", "d" * 200, "", "p") for i in range(60)]
        block = skills_block(skills)
        assert "more (call skill with a name" in block
        assert len(block) < 4096 + 300


class TestAssembly:
    def test_mode_memory_and_environment_sections(self, tmp_path):
        root = repo(tmp_path)
        prompt = build_system_prompt(PromptInputs(
            cwd=root, mode=Mode.PLAN, memory="- user prefers tabs", model="m", provider="p",
            skills=[Skill("s", "d", "", "p")],
        ))
        assert "Plan mode is active" in prompt
        assert "<memory>" in prompt and "user prefers tabs" in prompt
        assert f"Working directory: {root}" in prompt
        assert "Git branch: main" in prompt
        assert "Active model: m" in prompt
        assert "<available_skills>" in prompt

    def test_user_guidelines_precede_project_and_are_labeled(self, tmp_path):
        root = repo(tmp_path)
        (root / "AGENTS.md").write_text("PROJECT")
        userfile = tmp_path / "SUPERCLAW.md"
        userfile.write_text("PERSONAL")
        prompt = build_system_prompt(PromptInputs(cwd=root, mode=Mode.ASK, user_guidelines=userfile))
        assert "## User guidelines" in prompt
        assert prompt.index("PERSONAL") < prompt.index("PROJECT")
