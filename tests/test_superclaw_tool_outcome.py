import json
from pathlib import Path

from superclaw.loop import Options, run
from superclaw.policy import Mode, Policy
from superclaw.runtime import Completion, ToolCall
from superclaw.settings import LIMITS
from superclaw.tools import Registry, Result, Safety, SideEffect, Permission, Tool, ToolContext
from superclaw.tools.budget import OMISSION, Budget, Category, budget_output
from superclaw.tools.files import core_file_tools
from superclaw.tools.shell import Bash
from superclaw.tools.spill import SpillStore

TINY = Budget(max_tokens=60, max_chars=400)


def numbered(n):
    return "\n".join(f"line {i}" for i in range(n))


def test_file_budget_keeps_both_ends_and_marks_the_gap():
    out = budget_output(numbered(400), Category.FILE, TINY)
    assert out.truncated and out.reason == "semantic_file_budget"
    assert out.text.startswith("line 0\nline 1") and out.text.endswith("line 399")
    assert OMISSION.split("{")[0] in out.text and out.retained_tokens <= TINY.max_tokens


def test_test_budget_keeps_failure_context_and_tail():
    lines = [f"ok {i}" for i in range(300)]
    lines[150] = "FAILED tests/test_x.py::test_y - AssertionError"
    out = budget_output("\n".join(lines), Category.TEST, TINY)
    assert "FAILED tests/test_x.py" in out.text and "ok 148" in out.text and "ok 299" in out.text
    assert "ok 20" not in out.text


def test_search_budget_keeps_one_hit_per_file():
    hits = [f"src/{'a' if i < 200 else 'b'}.py:{i}:match here" for i in range(220)]
    out = budget_output("\n".join(hits), Category.SEARCH, TINY)
    assert "src/a.py:0:" in out.text and "src/b.py:200:" in out.text


def test_unfittable_output_falls_back_to_head_and_tail():
    out = budget_output("x" * 5000, Category.PROCESS, TINY)
    assert out.reason == "head_tail_budget" and out.text.startswith("xxx") and out.text.endswith("xxx") and len(out.text) <= TINY.max_chars


class Big(Tool):
    name = "big"
    description = "Return a huge string."
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}
    safety = Safety(SideEffect.NONE, Permission.ALLOW, "No side effects.")

    def run(self, args, ctx):
        return Result.success("\n".join(f"secret_token=abc{i}" for i in range(20_000)))


def test_registry_spills_the_full_redacted_output_and_reports_honestly(tmp_path):
    reg = Registry(spill=SpillStore(tmp_path / "artifacts"))
    reg.register(Big())
    res = reg.run("big", {}, ToolContext(workspace=tmp_path, session_id="s1"), call_id="call_9")
    saved = (tmp_path / "artifacts" / "s1" / "big-call_9.txt").read_text()
    assert res.artifact.path.endswith("big-call_9.txt") and res.artifact.complete
    assert "abc1\n" not in saved and "[REDACTED]" in saved and saved.count("\n") == 19_999
    assert res.diagnostics.model_tokens <= LIMITS.tool_output_tokens and res.diagnostics.redacted
    assert f"full output saved to {res.artifact.path}" in res.output


def test_hook_feedback_is_redacted_and_rebudgeted(tmp_path, monkeypatch):
    from superclaw.hooks import Dispatcher, Hook

    reg = Registry(spill=SpillStore(tmp_path / "artifacts"))
    for t in core_file_tools():
        reg.register(t)
    (tmp_path / "a.txt").write_text("hello\n")
    script = tmp_path / "hook.py"
    script.write_text("import json; print(json.dumps({'additionalContext': 'api_key=sk-proj-abcdefghijklmnopqrstuvwxyz ' + 'z' * 300000}))")
    hooks = Dispatcher([Hook(id="h", event="afterTool", command=[sys_executable(), str(script)])], tmp_path)
    turn = Completion(tool_calls=[ToolCall("c1", "read_file", json.dumps({"path": "a.txt"}))])
    events = []
    run("go", Scripted(turn, Completion(text="done")), Options(registry=reg, policy=Policy(tmp_path, Mode.AUTO), workspace=tmp_path,
                                                                system_prompt="S", hooks=hooks, on_event=events.append))
    result = next(e for e in events if e["type"] == "tool_result")
    saved = Path(result["artifact"]).read_text()
    assert "sk-proj-" not in result["output"] and "sk-proj-" not in saved
    assert "[REDACTED]" in saved and saved.endswith("z" * 100)
    assert result["diagnostics"]["truncated"] and result["diagnostics"]["model_tokens"] <= LIMITS.tool_output_tokens


def test_read_file_pages_and_clips_long_lines(tmp_path):
    reg = Registry()
    for t in core_file_tools():
        reg.register(t)
    (tmp_path / "big.txt").write_text("\n".join(["x" * 3000] + [f"l{i}" for i in range(LIMITS.read_file_lines + 5)]))
    res = reg.run("read_file", {"path": "big.txt"}, ToolContext(workspace=tmp_path))
    assert res.output.count("\n") == LIMITS.read_file_lines
    assert "… [+1,000 chars]" in res.output.split("\n")[0]
    assert f"[6 more lines; call read_file with offset={LIMITS.read_file_lines + 1} to continue]" in res.output


def test_edits_carry_a_diff_preview_the_model_never_sees(tmp_path):
    reg = Registry()
    for t in core_file_tools():
        reg.register(t)
    ctx = ToolContext(workspace=tmp_path)
    (tmp_path / "a.py").write_text("return a - b\n")
    reg.run("read_file", {"path": "a.py"}, ctx)
    res = reg.run("edit_file", {"path": "a.py", "description": "fix", "old_string": "a - b", "new_string": "a + b"}, ctx)
    assert res.output == "Replaced 1 occurrence(s) in a.py"
    assert res.display.kind == "diff" and "-return a - b\n+return a + b" in res.display.preview


def test_grep_skips_binary_and_oversized_files(tmp_path):
    reg = Registry()
    for t in core_file_tools():
        reg.register(t)
    (tmp_path / "bin.dat").write_bytes(b"needle\0\0\0")
    (tmp_path / "huge.txt").write_text("needle\n" + "x" * (LIMITS.read_file_bytes + 1))
    (tmp_path / "ok.txt").write_text("needle\n")
    res = reg.run("grep", {"pattern": "needle"}, ToolContext(workspace=tmp_path))
    assert res.output == "ok.txt:1:needle"


def test_bash_output_category_follows_the_command():
    bash = Bash()
    assert bash.category({"command": "pytest -q"}) is Category.TEST
    assert bash.category({"command": "ls -la"}) is Category.PROCESS


class Scripted:
    def __init__(self, *completions):
        self.queue = list(completions)

    def complete(self, messages, tools):
        return self.queue.pop(0) if self.queue else Completion(text="(exhausted)")


def sys_executable():
    import sys

    return sys.executable
