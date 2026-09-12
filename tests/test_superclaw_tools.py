import os

import pytest

from superclaw.redaction import REDACTED, redact
from superclaw.sandbox import Bubblewrap, Grant, detect
from superclaw.settings import LIMITS
from superclaw.skills import Skill, load_skills
from superclaw.tools import PathEscapes, Permission, Registry, Result, Safety, SideEffect, Tool, ToolContext, jail
from superclaw.tools.ask import NON_INTERACTIVE_MESSAGE, AskUser
from superclaw.tools.budget import OMISSION, Budget, Category, budget_output
from superclaw.tools.files import core_file_tools
from superclaw.tools.plan import UpdatePlan, pending_items
from superclaw.tools.shell import Bash
from superclaw.tools.skill import SkillTool
from superclaw.tools.spill import SpillStore

TINY = Budget(max_tokens=60, max_chars=400)


class Echo(Tool):
    name = "echo"
    description = "Echo the text back."
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": False}
    safety = Safety(SideEffect.NONE, Permission.ALLOW, "No side effects.")

    def run(self, args, ctx):
        if args["text"] == "boom":
            raise RuntimeError("kaboom")
        return Result.success(args["text"])


class Leaky(Tool):
    name = "leaky"
    description = "Return many secrets."
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}
    safety = Safety(SideEffect.NONE, Permission.ALLOW, "No side effects.")

    def run(self, args, ctx):
        return Result.success("\n".join(f"secret_token=abc{i}" for i in range(20_000)))


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("alpha\nbeta\ngamma\n")
    (tmp_path / "src" / "b.txt").write_text("Beta here\nand beta again\n")
    (tmp_path / "README.md").write_text("# readme\n")
    return tmp_path


@pytest.fixture
def ctx(ws):
    return ToolContext(workspace=ws)


@pytest.fixture
def reg():
    reg = Registry()
    for tool in core_file_tools():
        reg.register(tool)
    return reg


def test_registry_exposes_openai_shapes_and_turns_failures_into_error_results(ctx):
    reg = Registry()
    reg.register(Echo())
    assert reg.definitions() == [{"type": "function", "function": {"name": "echo", "description": "Echo the text back.", "parameters": Echo.parameters}}]
    assert reg.definitions(lambda t: t.name != "echo") == []
    assert reg.run("echo", {"text": "hi"}, ctx).output == "hi"
    assert "unknown tool" in reg.run("nope", {}, ctx).output
    assert "kaboom" in reg.run("echo", {"text": "boom"}, ctx).output


def test_jail_admits_inside_paths_and_refuses_every_escape(ws, tmp_path_factory):
    other = tmp_path_factory.mktemp("other")
    (other / "secret").write_text("s")
    os.symlink(other / "secret", ws / "link")
    assert jail(ws, "src/a.py") == (ws / "src" / "a.py").resolve()
    assert jail(ws, str(ws / "README.md")) == (ws / "README.md").resolve()
    for escape in ("../outside.txt", str(other / "x"), "link"):
        with pytest.raises(PathEscapes):
            jail(ws, escape)


def test_read_file_numbers_pages_and_clips_lines(reg, ctx, ws):
    assert reg.run("read_file", {"path": "src/a.py"}, ctx).output == "1→alpha\n2→beta\n3→gamma"
    assert reg.run("read_file", {"path": "src/a.py", "offset": 2, "limit": 1}, ctx).output == "2→beta\n[1 more lines; call read_file with offset=3 to continue]"
    assert "not found" in reg.run("read_file", {"path": "src/zzz.py"}, ctx).output
    assert "escapes" in reg.run("read_file", {"path": "../x"}, ctx).output
    (ws / "big.txt").write_text("\n".join(["x" * 3000] + [f"l{i}" for i in range(LIMITS.read_file_lines + 5)]))
    res = reg.run("read_file", {"path": "big.txt"}, ctx)
    assert res.output.count("\n") == LIMITS.read_file_lines and "… [+1,000 chars]" in res.output.split("\n")[0]
    assert f"[6 more lines; call read_file with offset={LIMITS.read_file_lines + 1} to continue]" in res.output


def test_writes_need_a_read_detect_drift_and_carry_a_diff_the_model_never_sees(reg, ctx, ws):
    res = reg.run("write_file", {"path": "new/f.txt", "description": "d", "content": "x\n"}, ctx)
    assert res.ok and (ws / "new" / "f.txt").read_text() == "x\n" and res.changed_files == ["new/f.txt"]
    assert not reg.run("write_file", {"path": "README.md", "description": "d", "content": "gone"}, ctx).ok
    edit = {"path": "src/a.py", "description": "d", "old_string": "beta", "new_string": "BETA"}
    assert "read the file before" in reg.run("edit_file", edit, ctx).output
    reg.run("read_file", {"path": "src/a.py"}, ctx)
    res = reg.run("edit_file", edit, ctx)
    assert res.output == "Replaced 1 occurrence(s) in src/a.py" and (ws / "src" / "a.py").read_text() == "alpha\nBETA\ngamma\n"
    assert res.display.kind == "diff" and "-beta\n+BETA" in res.display.preview
    assert "not found" in reg.run("edit_file", {**edit, "old_string": "delta"}, ctx).output
    assert "4 times" in reg.run("edit_file", {**edit, "old_string": "a", "new_string": ""}, ctx).output
    (ws / "src" / "a.py").write_text("changed outside\n")
    assert "changed on disk" in reg.run("edit_file", {**edit, "old_string": "changed"}, ctx).output
    assert "read it again" in reg.run("write_file", {"path": "src/a.py", "description": "d", "content": "x", "overwrite": True}, ctx).output


def test_listing_and_search_tools_skip_binaries_and_oversized_files(reg, ctx, ws):
    (ws / "bin.dat").write_bytes(b"beta\0\0")
    (ws / "huge.txt").write_text("beta\n" + "x" * (LIMITS.read_file_bytes + 1))
    assert reg.run("list_directory", {"path": "."}, ctx).output.splitlines() == ["README.md", "bin.dat", "huge.txt", "src/"]
    assert reg.run("list_directory", {"path": "src", "recursive": True}, ctx).output.splitlines() == ["a.py", "b.txt"]
    assert reg.run("glob", {"pattern": "**/*.py"}, ctx).output.splitlines() == ["src/a.py"]
    assert reg.run("grep", {"pattern": "beta", "case_insensitive": True}, ctx).output.splitlines() == ["src/a.py:2:beta", "src/b.txt:1:Beta here", "src/b.txt:2:and beta again"]
    assert reg.run("grep", {"pattern": "beta", "output_mode": "files_with_matches"}, ctx).output.splitlines() == ["src/a.py", "src/b.txt"]
    assert reg.run("grep", {"pattern": "beta", "output_mode": "count"}, ctx).output.splitlines() == ["src/a.py:1", "src/b.txt:1"]
    assert reg.run("grep", {"pattern": "beta", "glob": "*.txt"}, ctx).output.splitlines() == ["src/b.txt:2:and beta again"]
    for name in ("read_file", "list_directory", "glob", "grep"):
        assert (reg.get(name).safety.side_effect, reg.get(name).safety.permission) == (SideEffect.READ, Permission.ALLOW)
    for name in ("write_file", "edit_file"):
        assert (reg.get(name).safety.side_effect, reg.get(name).safety.permission) == (SideEffect.WRITE, Permission.PROMPT)


def test_semantic_budget_keeps_what_matters_per_category():
    numbered = "\n".join(f"line {i}" for i in range(400))
    out = budget_output(numbered, Category.FILE, TINY)
    assert out.reason == "semantic_file_budget" and out.text.startswith("line 0\nline 1") and out.text.endswith("line 399")
    assert OMISSION.split("{")[0] in out.text and out.retained_tokens <= TINY.max_tokens
    tests = [f"ok {i}" for i in range(300)]
    tests[150] = "FAILED tests/test_x.py::test_y - AssertionError"
    out = budget_output("\n".join(tests), Category.TEST, TINY)
    assert "FAILED tests/test_x.py" in out.text and "ok 148" in out.text and "ok 299" in out.text and "ok 100" not in out.text
    hits = "\n".join(f"src/{'a' if i < 200 else 'b'}.py:{i}:match here" for i in range(220))
    out = budget_output(hits, Category.SEARCH, TINY)
    assert "src/a.py:0:" in out.text and "src/b.py:200:" in out.text
    out = budget_output("x" * 5000, Category.PROCESS, TINY)
    assert out.reason == "head_tail_budget" and out.text.startswith("xxx") and out.text.endswith("xxx") and len(out.text) <= TINY.max_chars


def test_boundary_redacts_budgets_spills_and_reports_honestly(tmp_path):
    reg = Registry(spill=SpillStore(tmp_path / "artifacts"))
    reg.register(Leaky())
    res = reg.run("leaky", {}, ToolContext(workspace=tmp_path, session_id="s1"), call_id="call_9")
    saved = (tmp_path / "artifacts" / "s1" / "leaky-call_9.txt").read_text()
    assert res.artifact.path.endswith("leaky-call_9.txt") and res.artifact.complete
    assert "abc1\n" not in saved and "[REDACTED]" in saved and saved.count("\n") == 19_999
    assert res.diagnostics.model_tokens <= LIMITS.tool_output_tokens and res.diagnostics.redacted
    assert f"full output saved to {res.artifact.path}" in res.output and res.diagnostics.original_chars == len(saved)
    bare = Registry()
    bare.register(Leaky())
    assert "not recoverable" in bare.run("leaky", {}, ToolContext(workspace=tmp_path)).output
    (tmp_path / ".env.example").write_text("OPENAI_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz0123456789\nAuthorization: Bearer abc.def\nDB_PASSWORD='hunter2'\nMAX_TOKENS=5\n")
    for tool in core_file_tools():
        bare.register(tool)
    out = bare.run("read_file", {"path": ".env.example"}, ToolContext(workspace=tmp_path)).output
    assert "sk-proj-" not in out and "hunter2" not in out and "abc.def" not in out and "MAX_TOKENS=5" in out
    assert redact("plain text")[1] is False and redact("token=ghp_" + "a" * 40)[0].endswith(REDACTED)


def test_bash_runs_inside_the_workspace_times_out_and_categorises(ctx, tmp_path):
    (tmp_path / "sub").mkdir()
    bash = Bash()
    assert bash.run({"command": "echo hi"}, ctx).output == "hi"
    res = bash.run({"command": "echo bad >&2; exit 3"}, ctx)
    assert not res.ok and "bad" in res.output and "[exit 3]" in res.output
    assert bash.run({"command": "pwd", "cwd": "sub"}, ctx).output == str((tmp_path / "sub").resolve())
    reg = Registry()
    reg.register(bash)
    assert "escapes" in reg.run("bash", {"command": "pwd", "cwd": "/"}, ctx).output
    assert "timed out" in bash.run({"command": "sleep 5", "timeout_ms": 200}, ctx).output
    assert bash.category({"command": "pytest -q"}) is Category.TEST and bash.category({"command": "ls"}) is Category.PROCESS
    argv = Bubblewrap().wrap(["bash", "-c", "x"], tmp_path, tmp_path, Grant(network=True, paths=["/opt/extra"]))
    assert argv[0] == "bwrap" and argv[-3:] == ["bash", "-c", "x"] and "--unshare-net" not in argv and "/opt/extra" in argv


@pytest.mark.skipif(detect() is None, reason="bwrap not installed")
def test_sandbox_blocks_network_and_outside_writes_but_allows_workspace(ctx):
    tool = Bash(detect())
    assert tool.run({"command": "echo in > made.txt && cat made.txt"}, ctx).output == "in"
    assert not tool.run({"command": "touch /etc/superclaw-probe"}, ctx).ok
    assert not tool.run({"command": "curl -sm2 https://example.com"}, ctx).ok
    assert tool.run({"command": "ls ~/.ssh 2>/dev/null | wc -l"}, ctx).output == "0"
    ctx.state["approval"] = {"escalated": True, "network": False}
    assert tool.run({"command": "test -w /tmp && echo host"}, ctx).output == "host"


def test_update_plan_normalises_statuses_and_reports_pending(ctx):
    res = UpdatePlan().run({"plan": [{"content": "read code"}, {"content": "edit", "status": "in_progress", "notes": "careful"}]}, ctx)
    assert res.output == "Current Plan:\n1. [pending] read code\n2. [in_progress] edit\n   Notes: careful"
    UpdatePlan().run({"plan": [{"content": "a", "status": "done"}, {"content": "b", "status": "todo"}, {"content": "c", "status": "wip"}, {"content": "d", "status": "in_progress"}]}, ctx)
    assert [i["status"] for i in ctx.state["plan"]] == ["completed", "pending", "completed", "in_progress"]
    assert pending_items(ctx.state) == ["b", "d"]
    assert UpdatePlan().run({"plan": []}, ctx).output == "Plan is currently empty."
    assert not UpdatePlan().run({"plan": [{"status": "pending"}]}, ctx).ok
    assert AskUser().run({"questions": [{"question": "Which?"}]}, ctx).output == NON_INTERACTIVE_MESSAGE
    assert not AskUser().run({}, ctx).ok


def _skill(root, name, description, body, frontmatter_name=None):
    (root / name).mkdir(parents=True)
    fm = f"---\ndescription: {description}\n" + (f"name: {frontmatter_name}\n" if frontmatter_name else "") + "---\n"
    (root / name / "SKILL.md").write_text(fm + body)


def test_skills_load_shadow_by_root_order_and_stay_inside_their_root(tmp_path):
    a, b, root = tmp_path / "a", tmp_path / "b", tmp_path / "root"
    _skill(a, "bench", "Run benchmarks.", "# Bench\n1. make bench\n")
    _skill(a, "dir", "d", "body", frontmatter_name="real-name")
    _skill(a, "s", "from a", "A")
    _skill(b, "s", "from b", "B")
    (a / "nofile").mkdir()
    (a / "plain.txt").write_text("x")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text("---\ndescription: x\n---\nleak")
    root.mkdir()
    (root / "evil").symlink_to(outside)
    skills = load_skills([tmp_path / "missing", a, b, root])
    assert [s.name for s in skills] == ["bench", "real-name", "s"] and [s.content for s in skills if s.name == "s"] == ["A"]
    assert skills[0] == Skill(name="bench", description="Run benchmarks.", content="# Bench\n1. make bench", path=str(a / "bench" / "SKILL.md"))
    tool = SkillTool(roots=[a])
    assert tool.run({"name": "s"}, ToolContext(workspace=tmp_path)).output == "A"
    assert "bench, real-name, s" in tool.run({"name": "zzz"}, ToolContext(workspace=tmp_path)).output
