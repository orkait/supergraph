import pytest

from superclaw.sandbox import Bubblewrap, Grant, detect
from superclaw.skills import Skill, load_skills
from superclaw.tools import Registry, ToolContext
from superclaw.tools.ask import AskUser, NON_INTERACTIVE_MESSAGE
from superclaw.tools.plan import UpdatePlan, pending_items
from superclaw.tools.shell import Bash
from superclaw.tools.skill import SkillTool


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(workspace=tmp_path)


class TestBash:
    def test_runs_command_and_returns_stdout(self, ctx):
        res = Bash().run({"command": "echo hi"}, ctx)
        assert res.ok
        assert res.output == "hi"

    def test_nonzero_exit_is_error_with_stderr_and_code(self, ctx):
        res = Bash().run({"command": "echo bad >&2; exit 3"}, ctx)
        assert not res.ok
        assert "bad" in res.output
        assert "[exit 3]" in res.output

    def test_cwd_is_jailed(self, ctx):
        reg = Registry()
        reg.register(Bash())
        res = reg.run("bash", {"command": "pwd", "cwd": "/"}, ctx)
        assert not res.ok
        assert "escapes" in res.output

    def test_cwd_relative_to_workspace(self, ctx, tmp_path):
        (tmp_path / "sub").mkdir()
        res = Bash().run({"command": "pwd", "cwd": "sub"}, ctx)
        assert res.output == str((tmp_path / "sub").resolve())

    def test_timeout(self, ctx):
        res = Bash().run({"command": "sleep 5", "timeout_ms": 200}, ctx)
        assert not res.ok
        assert "timed out" in res.output

    @pytest.mark.skipif(detect() is None, reason="bwrap not installed")
    def test_sandbox_blocks_network_and_outside_writes_but_allows_workspace(self, ctx, tmp_path):
        tool = Bash(detect())
        assert tool.run({"command": "echo in > made.txt && cat made.txt"}, ctx).output == "in"
        assert not tool.run({"command": "touch /etc/superclaw-probe"}, ctx).ok
        assert not tool.run({"command": "curl -sm2 https://example.com"}, ctx).ok
        assert tool.run({"command": "ls ~/.ssh 2>/dev/null | wc -l"}, ctx).output == "0"
        ctx.state["approval"] = {"escalated": True, "network": False}
        assert tool.run({"command": "test -w /tmp && echo host"}, ctx).output == "host"

    def test_wrap_argv_shape(self, tmp_path):
        argv = Bubblewrap().wrap(["bash", "-c", "x"], tmp_path, tmp_path, Grant(network=True, paths=["/opt/extra"]))
        assert argv[0] == "bwrap" and argv[-3:] == ["bash", "-c", "x"]
        assert "--unshare-net" not in argv
        assert argv[argv.index("--bind") + 1] == str(tmp_path)
        assert "/opt/extra" in argv


class TestUpdatePlan:
    def test_formats_and_stores_plan(self, ctx):
        res = UpdatePlan().run({"plan": [{"content": "read code"}, {"content": "edit", "status": "in_progress", "notes": "careful"}]}, ctx)
        assert res.ok
        assert res.output == "Current Plan:\n1. [pending] read code\n2. [in_progress] edit\n   Notes: careful"
        assert ctx.state["plan"][1]["status"] == "in_progress"

    def test_only_one_in_progress_earlier_become_completed(self, ctx):
        UpdatePlan().run({"plan": [{"content": "a", "status": "in_progress"}, {"content": "b", "status": "in_progress"}]}, ctx)
        assert [i["status"] for i in ctx.state["plan"]] == ["completed", "in_progress"]

    def test_unknown_status_is_coerced(self, ctx):
        UpdatePlan().run({"plan": [{"content": "a", "status": "done"}, {"content": "b", "status": "todo"}, {"content": "c", "status": "wip"}]}, ctx)
        assert [i["status"] for i in ctx.state["plan"]] == ["completed", "pending", "in_progress"]

    def test_empty_plan(self, ctx):
        assert UpdatePlan().run({"plan": []}, ctx).output == "Plan is currently empty."

    def test_missing_content_is_error(self, ctx):
        assert not UpdatePlan().run({"plan": [{"status": "pending"}]}, ctx).ok

    def test_pending_items_helper(self, ctx):
        UpdatePlan().run({"plan": [{"content": "a", "status": "completed"}, {"content": "b"}, {"content": "c", "status": "in_progress"}]}, ctx)
        assert pending_items(ctx.state) == ["b", "c"]


def _write_skill(root, name, description, body, frontmatter_name=None):
    d = root / name
    d.mkdir(parents=True)
    fm = f"---\ndescription: {description}\n" + (f"name: {frontmatter_name}\n" if frontmatter_name else "") + "---\n"
    (d / "SKILL.md").write_text(fm + body)
    return d


class TestSkills:
    def test_load_parses_frontmatter_and_body(self, tmp_path):
        _write_skill(tmp_path, "bench", "Run benchmarks.", "# Bench\n1. make bench\n")
        skills = load_skills([tmp_path])
        assert skills == [Skill(name="bench", description="Run benchmarks.", content="# Bench\n1. make bench", path=str(tmp_path / "bench" / "SKILL.md"))]

    def test_frontmatter_name_overrides_directory(self, tmp_path):
        _write_skill(tmp_path, "dir", "d", "body", frontmatter_name="real-name")
        assert load_skills([tmp_path])[0].name == "real-name"

    def test_missing_root_and_bad_dirs_are_skipped(self, tmp_path):
        (tmp_path / "nofile").mkdir()
        (tmp_path / "plain.txt").write_text("x")
        assert load_skills([tmp_path / "missing", tmp_path]) == []

    def test_earlier_root_wins_name_collision(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        _write_skill(a, "s", "from a", "A")
        _write_skill(b, "s", "from b", "B")
        skills = load_skills([a, b])
        assert [s.content for s in skills] == ["A"]

    def test_symlink_escaping_root_is_skipped(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "SKILL.md").write_text("---\ndescription: x\n---\nleak")
        root = tmp_path / "root"
        root.mkdir()
        (root / "evil").symlink_to(outside)
        assert load_skills([root]) == []

    def test_skill_tool_loads_by_name_and_lists_on_unknown(self, tmp_path):
        _write_skill(tmp_path, "one", "first", "ONE")
        _write_skill(tmp_path, "two", "second", "TWO")
        tool = SkillTool(roots=[tmp_path])
        ctx = ToolContext(workspace=tmp_path)
        assert tool.run({"name": "two"}, ctx).output == "TWO"
        res = tool.run({"name": "zzz"}, ctx)
        assert not res.ok
        assert "one, two" in res.output


class TestAskUser:
    def test_non_interactive_fallback(self, ctx):
        res = AskUser().run({"questions": [{"question": "Which?"}]}, ctx)
        assert res.ok
        assert res.output == NON_INTERACTIVE_MESSAGE

    def test_requires_questions(self, ctx):
        assert not AskUser().run({}, ctx).ok


def test_registry_definitions_include_all_core(tmp_path):
    reg = Registry()
    for t in (Bash(), UpdatePlan(), SkillTool(roots=[tmp_path]), AskUser()):
        reg.register(t)
    assert reg.names() == ["ask_user", "bash", "skill", "update_plan"]
