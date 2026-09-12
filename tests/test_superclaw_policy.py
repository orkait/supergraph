import pytest

from superclaw.policy import Action, Mode, Policy, classify_command, validate_prefix
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


def test_command_classes_and_prefix_rules():
    assert all("destructive" in classify_command(c).categories for c in ("rm -rf build", "sudo apt update", "git push --force", "docker system prune", "ls && rm -rf /tmp/x"))
    assert all("network" in classify_command(c).categories for c in ("curl https://x", "git pull", "pip install x", "npm i x", "gh pr create"))
    assert all("interactive" in classify_command(c).categories for c in ("vim a.py", "less log", "python", "git rebase -i HEAD~3"))
    assert all(classify_command(c).categories == [] for c in ("ls -la", "pytest -q", "git status", "python -c 'print(1)'", "rm x.tmp"))
    assert validate_prefix(["git", "pull"], "git pull origin main") == ""
    assert "too broad" in validate_prefix(["rm", "-rf"], "rm -rf x") and "single-token" in validate_prefix(["make"], "make test")
    assert "heredoc" in validate_prefix(["cat", "-n"], "cat -n <<EOF\nx\nEOF") and "match the start" in validate_prefix(["git", "push"], "git pull")
