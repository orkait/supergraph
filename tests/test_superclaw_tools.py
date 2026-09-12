import os

import pytest

from superclaw.tools import (
    PathEscapes,
    Permission,
    Registry,
    Result,
    Safety,
    SideEffect,
    Tool,
    ToolContext,
    jail,
)
from superclaw.tools.files import core_file_tools


class Echo(Tool):
    name = "echo"
    description = "Echo the text back."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.NONE, Permission.ALLOW, "No side effects.")

    def run(self, args, ctx):
        return Result.success(args["text"])


class Big(Tool):
    name = "big"
    description = "Return a huge string."
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}
    safety = Safety(SideEffect.NONE, Permission.ALLOW, "No side effects.")

    def run(self, args, ctx):
        return Result.success("x" * 200_000)


class Boom(Tool):
    name = "boom"
    description = "Raise."
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}
    safety = Safety(SideEffect.NONE, Permission.ALLOW, "No side effects.")

    def run(self, args, ctx):
        raise RuntimeError("kaboom")


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


class TestRegistry:
    def test_definitions_use_openai_function_shape(self):
        reg = Registry()
        reg.register(Echo())
        defs = reg.definitions()
        assert defs == [{
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo the text back.",
                "parameters": Echo.parameters,
            },
        }]

    def test_definitions_can_be_filtered(self):
        reg = Registry()
        reg.register(Echo())
        assert reg.definitions(lambda t: t.name != "echo") == []

    def test_run_unknown_tool_is_an_error_result(self, ctx):
        reg = Registry()
        res = reg.run("nope", {}, ctx)
        assert not res.ok
        assert "unknown tool" in res.output

    def test_run_dispatches_to_tool(self, ctx):
        reg = Registry()
        reg.register(Echo())
        assert reg.run("echo", {"text": "hi"}, ctx).output == "hi"

    def test_get_returns_tool_or_none(self):
        reg = Registry()
        reg.register(Echo())
        assert reg.get("echo").name == "echo"
        assert reg.get("missing") is None

    def test_run_truncates_oversized_output(self, ctx):
        reg = Registry()
        reg.register(Big())
        res = reg.run("big", {}, ctx)
        assert res.ok
        assert res.truncated
        assert len(res.output) < 70_000
        assert "truncated" in res.output

    def test_run_turns_tool_exception_into_error_result(self, ctx):
        reg = Registry()
        reg.register(Boom())
        res = reg.run("boom", {}, ctx)
        assert not res.ok
        assert "kaboom" in res.output


class TestJail:
    def test_relative_path_resolves_inside_workspace(self, ws):
        assert jail(ws, "src/a.py") == (ws / "src" / "a.py").resolve()

    def test_dot_dot_escape_is_refused(self, ws):
        with pytest.raises(PathEscapes):
            jail(ws, "../outside.txt")

    def test_absolute_path_inside_is_accepted(self, ws):
        assert jail(ws, str(ws / "README.md")) == (ws / "README.md").resolve()

    def test_absolute_path_outside_is_refused(self, ws, tmp_path_factory):
        other = tmp_path_factory.mktemp("other")
        with pytest.raises(PathEscapes):
            jail(ws, str(other / "x"))

    def test_symlink_pointing_outside_is_refused(self, ws, tmp_path_factory):
        other = tmp_path_factory.mktemp("other")
        (other / "secret").write_text("s")
        os.symlink(other / "secret", ws / "link")
        with pytest.raises(PathEscapes):
            jail(ws, "link")


class TestFileTools:
    @pytest.fixture
    def reg(self):
        reg = Registry()
        for tool in core_file_tools():
            reg.register(tool)
        return reg

    def test_read_file_numbers_lines(self, reg, ctx):
        res = reg.run("read_file", {"path": "src/a.py"}, ctx)
        assert res.ok
        assert res.output == "1→alpha\n2→beta\n3→gamma"

    def test_read_file_offset_and_limit(self, reg, ctx):
        res = reg.run("read_file", {"path": "src/a.py", "offset": 2, "limit": 1}, ctx)
        assert res.output == "2→beta"

    def test_read_file_missing_is_error(self, reg, ctx):
        res = reg.run("read_file", {"path": "src/zzz.py"}, ctx)
        assert not res.ok
        assert "not found" in res.output

    def test_read_file_outside_workspace_is_error(self, reg, ctx):
        res = reg.run("read_file", {"path": "../x"}, ctx)
        assert not res.ok
        assert "escapes" in res.output

    def test_write_file_creates_and_reports_change(self, reg, ctx, ws):
        res = reg.run("write_file", {"path": "new/dir/f.txt", "content": "x\n"}, ctx)
        assert res.ok
        assert (ws / "new" / "dir" / "f.txt").read_text() == "x\n"
        assert res.changed_files == ["new/dir/f.txt"]

    def test_write_file_refuses_overwrite_by_default(self, reg, ctx, ws):
        res = reg.run("write_file", {"path": "README.md", "content": "gone"}, ctx)
        assert not res.ok
        assert (ws / "README.md").read_text() == "# readme\n"

    def test_write_file_overwrite_when_asked(self, reg, ctx, ws):
        reg.run("read_file", {"path": "README.md"}, ctx)
        res = reg.run("write_file", {"path": "README.md", "content": "new", "overwrite": True}, ctx)
        assert res.ok
        assert (ws / "README.md").read_text() == "new"

    def test_edit_file_replaces_unique_match(self, reg, ctx, ws):
        reg.run("read_file", {"path": "src/a.py"}, ctx)
        res = reg.run("edit_file", {"path": "src/a.py", "old_string": "beta", "new_string": "BETA"}, ctx)
        assert res.ok
        assert (ws / "src" / "a.py").read_text() == "alpha\nBETA\ngamma\n"
        assert res.changed_files == ["src/a.py"]

    def test_edit_file_not_found_is_error(self, reg, ctx):
        reg.run("read_file", {"path": "src/a.py"}, ctx)
        res = reg.run("edit_file", {"path": "src/a.py", "old_string": "delta", "new_string": "x"}, ctx)
        assert not res.ok
        assert "not found" in res.output

    def test_edit_file_ambiguous_requires_replace_all(self, reg, ctx, ws):
        reg.run("read_file", {"path": "src/a.py"}, ctx)
        res = reg.run("edit_file", {"path": "src/a.py", "old_string": "a", "new_string": ""}, ctx)
        assert not res.ok
        assert "5 times" in res.output
        res = reg.run("edit_file", {"path": "src/a.py", "old_string": "a", "new_string": "", "replace_all": True}, ctx)
        assert res.ok
        assert (ws / "src" / "a.py").read_text() == "lph\nbet\ngmm\n"

    def test_list_directory_flat(self, reg, ctx):
        res = reg.run("list_directory", {"path": "."}, ctx)
        assert res.output.splitlines() == ["README.md", "src/"]

    def test_list_directory_recursive(self, reg, ctx):
        res = reg.run("list_directory", {"path": ".", "recursive": True}, ctx)
        assert res.output.splitlines() == ["README.md", "src/", "src/a.py", "src/b.txt"]

    def test_glob_matches_pattern(self, reg, ctx):
        res = reg.run("glob", {"pattern": "**/*.py"}, ctx)
        assert res.output.splitlines() == ["src/a.py"]

    def test_grep_content_mode(self, reg, ctx):
        res = reg.run("grep", {"pattern": "beta", "case_insensitive": True}, ctx)
        assert res.output.splitlines() == [
            "src/a.py:2:beta",
            "src/b.txt:1:Beta here",
            "src/b.txt:2:and beta again",
        ]

    def test_grep_files_with_matches(self, reg, ctx):
        res = reg.run("grep", {"pattern": "beta", "output_mode": "files_with_matches"}, ctx)
        assert res.output.splitlines() == ["src/a.py", "src/b.txt"]

    def test_grep_count(self, reg, ctx):
        res = reg.run("grep", {"pattern": "beta", "output_mode": "count"}, ctx)
        assert res.output.splitlines() == ["src/a.py:1", "src/b.txt:1"]

    def test_grep_glob_filter(self, reg, ctx):
        res = reg.run("grep", {"pattern": "beta", "glob": "*.txt"}, ctx)
        assert res.output.splitlines() == ["src/b.txt:2:and beta again"]

    def test_read_tools_are_allow_read(self, reg):
        for name in ("read_file", "list_directory", "glob", "grep"):
            s = reg.get(name).safety
            assert (s.side_effect, s.permission) == (SideEffect.READ, Permission.ALLOW)

    def test_write_tools_are_prompt_write(self, reg):
        for name in ("write_file", "edit_file"):
            s = reg.get(name).safety
            assert (s.side_effect, s.permission) == (SideEffect.WRITE, Permission.PROMPT)
