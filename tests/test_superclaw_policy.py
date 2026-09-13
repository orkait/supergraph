import pytest

from superclaw.cli import _tool_set, build_parser
from superclaw.policy import Action, Mode, Policy, classify_command, next_mode, validate_prefix
from superclaw.schema import SchemaError
from superclaw.schema import extract as schema_extract
from superclaw.schema import instruction as schema_instruction
from superclaw.schema import problems as schema_problems
from superclaw.settings import Settings
from superclaw.tools import Permission, Registry, Safety, SideEffect, Tool
from superclaw.tools.files import core_file_tools
from superclaw.tools.shell import Bash


class Fetch(Tool):
    name = "web_fetch"
    description = "Fetch a URL."
    parameters = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
    safety = Safety(SideEffect.NETWORK, Permission.PROMPT, "Network egress.")

    def run(self, args, ctx):
        raise NotImplementedError


@pytest.fixture
def reg():
    reg = Registry()
    for t in (*core_file_tools(), Bash(), Fetch()):
        reg.register(t)
    return reg


def by_mode(reg, ws, tool, args, sandboxed=True):
    return {m.value: Policy(ws, m, sandboxed=sandboxed).evaluate(reg.get(tool), args).action for m in Mode}


def test_modes_shell_risk_and_grants(reg, tmp_path):
    expected = {"ask": Action.PROMPT, "auto": Action.ALLOW, "plan": Action.DENY, "unsafe": Action.ALLOW}
    assert by_mode(reg, tmp_path, "write_file", {"path": "a", "content": ""}) == expected and by_mode(reg, tmp_path, "bash", {"command": "ls"}) == expected
    assert set(by_mode(reg, tmp_path, "read_file", {"path": "../x"}).values()) == {Action.DENY}
    shared = tmp_path.parent / "shared"
    write = reg.get("write_file")
    assert Policy(tmp_path, Mode.AUTO, sandboxed=True).evaluate(write, {"path": str(shared / "a")}).action == Action.DENY
    assert Policy(tmp_path, Mode.AUTO, sandboxed=True, extra_dirs=(shared,)).evaluate(write, {"path": str(shared / "a")}).action == Action.ALLOW
    assert by_mode(reg, tmp_path, "web_fetch", {"url": "https://x"})["auto"] == Action.PROMPT
    destructive = by_mode(reg, tmp_path, "bash", {"command": "rm -rf build"})
    assert (destructive["auto"], destructive["unsafe"]) == (Action.PROMPT, Action.ALLOW)
    auto = Policy(tmp_path, Mode.AUTO, sandboxed=True)
    assert auto.evaluate(reg.get("bash"), {"command": "pip install requests"}).action == Action.PROMPT
    assert auto.evaluate(reg.get("bash"), {"command": "ls", "sandbox_permissions": "require_escalated"}).action == Action.DENY
    assert Policy(tmp_path, Mode.UNSAFE).evaluate(reg.get("bash"), {"command": "vim x"}).action == Action.DENY
    assert by_mode(reg, tmp_path, "bash", {"command": "ls"}, sandboxed=False)["auto"] == Action.PROMPT
    p = Policy(tmp_path, Mode.ASK)
    p.grant_session("bash")
    p.grant_prefix(["git", "pull"])
    assert p.evaluate(reg.get("bash"), {"command": "ls"}).action == Action.ALLOW and p.evaluate(reg.get("bash"), {"command": "rm -rf x"}).action == Action.PROMPT
    assert p.evaluate(reg.get("bash"), {"command": "git pull origin main"}).action == Action.ALLOW and p.evaluate(reg.get("bash"), {"command": "git push"}).action == Action.PROMPT
    assert [d["function"]["name"] for d in reg.definitions(Policy(tmp_path, Mode.PLAN).visible)] == ["glob", "grep", "list_directory", "read_file"]
    allow = Policy(tmp_path, Mode.AUTO, allow_tools=frozenset({"read_file", "grep"}))
    assert {d["function"]["name"] for d in reg.definitions(allow.visible)} == {"read_file", "grep"}
    names = {d["function"]["name"] for d in reg.definitions(Policy(tmp_path, Mode.AUTO, deny_tools=frozenset({"bash", "web_fetch"})).visible)}
    assert "bash" not in names and "web_fetch" not in names and "read_file" in names
    assert _tool_set("read_file, grep bash") == frozenset({"read_file", "grep", "bash"}) and _tool_set("") == frozenset()
    scoped = Policy(tmp_path, Mode.AUTO, allow_tools=frozenset({"read_file", "grep"}))
    scoped.scope_to(frozenset({"grep", "bash"}))
    assert scoped.allow_tools == frozenset({"grep"})
    scoped.scope_to(frozenset())
    assert scoped.allow_tools == frozenset({"read_file", "grep"})
    wide = Policy(tmp_path, Mode.AUTO)
    wide.scope_to(frozenset({"grep"}))
    assert wide.allow_tools == frozenset({"grep"})
    assert [next_mode(m) for m in (Mode.ASK, Mode.AUTO, Mode.PLAN, Mode.UNSAFE)] == [Mode.AUTO, Mode.PLAN, Mode.ASK, Mode.ASK]
    parsed = build_parser(Settings.from_env({})).parse_args(["--dangerously-skip-permissions", "exec", "x"])
    assert parsed.dangerously_skip_permissions and (Mode.UNSAFE.value if parsed.dangerously_skip_permissions else parsed.mode) == "unsafe"
    flags = build_parser(Settings.from_env({})).parse_args(["-w", "--add-dir", "/tmp", "exec", "x"])
    assert flags.worktree == "" and flags.add_dir == ["/tmp"]
    assert build_parser(Settings.from_env({})).parse_args(["--worktree", "alpha", "exec", "x"]).worktree == "alpha"
    reviewed = build_parser(Settings.from_env({})).parse_args(["review", "--base", "main", "focus on tests"])
    assert reviewed.base == "main" and reviewed.prompt == "focus on tests" and not reviewed.uncommitted
    with pytest.raises(SystemExit):
        build_parser(Settings.from_env({})).parse_args(["review", "--base", "main", "--commit", "abc"])


def test_command_classes_and_prefix_rules():
    assert all("destructive" in classify_command(c).categories for c in ("rm -rf build", "sudo apt update", "git push --force", "docker system prune", "ls && rm -rf /tmp/x"))
    assert all("network" in classify_command(c).categories for c in ("curl https://x", "git pull", "pip install x", "npm i x", "gh pr create"))
    assert all("interactive" in classify_command(c).categories for c in ("vim a.py", "less log", "python", "git rebase -i HEAD~3"))
    assert all(classify_command(c).categories == [] for c in ("ls -la", "pytest -q", "git status", "python -c 'print(1)'", "rm x.tmp"))
    assert validate_prefix(["git", "pull"], "git pull origin main") == ""
    assert "too broad" in validate_prefix(["rm", "-rf"], "rm -rf x") and "single-token" in validate_prefix(["make"], "make test")
    assert "heredoc" in validate_prefix(["cat", "-n"], "cat -n <<EOF\nx\nEOF") and "match the start" in validate_prefix(["git", "push"], "git pull")
    shape = {"type": "object", "required": ["verdict", "findings"],
             "properties": {"verdict": {"type": "string"}, "count": {"type": "integer"},
                            "findings": {"type": "array", "items": {"type": "object", "required": ["file"],
                                                                    "properties": {"file": {"type": "string"}}}}}}
    good = {"verdict": "ok", "count": 2, "findings": [{"file": "a.py"}]}
    assert schema_problems(good, shape) == [] and schema_extract('```json\n{"verdict": "ok"}\n```') == {"verdict": "ok"}
    assert schema_problems({"verdict": 1, "findings": [{"line": 2}]}, shape) == [
        "$.verdict: expected string, got int", "$.findings[0]: missing required property 'file'"]
    assert schema_problems({"verdict": "ok", "findings": [], "count": True}, shape) == ["$.count: expected integer, got boolean"]
    assert schema_problems([], shape) == ["$: expected object, got list"]
    with pytest.raises(SchemaError):
        schema_extract("sorry, no JSON here")
    assert "Your final message must be one JSON value" in schema_instruction(shape)
