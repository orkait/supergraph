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


class Denied(Tool):
    name = "denied"
    description = "Always denied."
    parameters = {"type": "object", "properties": {}}
    safety = Safety(SideEffect.NONE, Permission.DENY, "Disabled by policy.")

    def run(self, args, ctx):
        raise NotImplementedError


@pytest.fixture
def reg():
    reg = Registry()
    for t in (*core_file_tools(), Bash(), Fetch(), Denied()):
        reg.register(t)
    return reg


def by_mode(reg, ws, tool, args, sandboxed=True):
    return {m.value: Policy(ws, m, sandboxed=sandboxed).evaluate(reg.get(tool), args).action for m in Mode}


def test_modes_decide_writes_and_shell_while_reads_escapes_and_denied_tools_are_fixed(reg, tmp_path):
    expected = {"ask": Action.PROMPT, "auto": Action.ALLOW, "plan": Action.DENY, "unsafe": Action.ALLOW}
    assert by_mode(reg, tmp_path, "write_file", {"path": "a", "content": ""}) == expected
    assert by_mode(reg, tmp_path, "bash", {"command": "ls"}) == expected
    assert set(by_mode(reg, tmp_path, "read_file", {"path": "a"}).values()) == {Action.ALLOW}
    assert set(by_mode(reg, tmp_path, "read_file", {"path": "../x"}).values()) == {Action.DENY}
    assert set(by_mode(reg, tmp_path, "edit_file", {"path": "/etc/hosts", "old_string": "a", "new_string": "b"}).values()) == {Action.DENY}
    assert set(by_mode(reg, tmp_path, "denied", {}).values()) == {Action.DENY}
    fetch = by_mode(reg, tmp_path, "web_fetch", {"url": "https://x"})
    assert (fetch["auto"], fetch["unsafe"]) == (Action.PROMPT, Action.ALLOW)
    assert [d["function"]["name"] for d in reg.definitions(Policy(tmp_path, Mode.PLAN).visible)] == ["glob", "grep", "list_directory", "read_file"]
    assert [d["function"]["name"] for d in reg.definitions(Policy(tmp_path, Mode.ASK).visible)] == [n for n in reg.names() if n != "denied"]


def test_risky_shell_escalation_and_missing_sandbox_prompt_and_grants_narrow_it(reg, tmp_path):
    destructive = by_mode(reg, tmp_path, "bash", {"command": "rm -rf build"})
    assert (destructive["ask"], destructive["auto"], destructive["unsafe"]) == (Action.PROMPT, Action.PROMPT, Action.ALLOW)
    auto = Policy(tmp_path, Mode.AUTO, sandboxed=True)
    assert auto.evaluate(reg.get("bash"), {"command": "pip install requests"}).action == Action.PROMPT
    assert auto.evaluate(reg.get("bash"), {"command": "rm -rf build"}).risk.level == "critical"
    assert Policy(tmp_path, Mode.UNSAFE).evaluate(reg.get("bash"), {"command": "vim x"}).action == Action.DENY
    assert Policy(tmp_path, Mode.UNSAFE).evaluate(reg.get("bash"), {"command": "ls", "cwd": "/"}).action == Action.DENY
    assert by_mode(reg, tmp_path, "bash", {"command": "ls", "sandbox_permissions": "require_escalated", "justification": "need host state"})["auto"] == Action.PROMPT
    assert auto.evaluate(reg.get("bash"), {"command": "ls", "sandbox_permissions": "require_escalated"}).action == Action.DENY
    no_backend = by_mode(reg, tmp_path, "bash", {"command": "ls"}, sandboxed=False)
    assert (no_backend["auto"], no_backend["unsafe"]) == (Action.PROMPT, Action.ALLOW)
    d = auto.evaluate(reg.get("bash"), {"command": "curl x", "additional_permissions": {"network": True}})
    assert (d.action, d.network, d.escalated) == (Action.PROMPT, True, False)
    p = Policy(tmp_path, Mode.ASK)
    p.grant_session("bash")
    assert p.evaluate(reg.get("bash"), {"command": "ls"}).action == Action.ALLOW and p.evaluate(reg.get("bash"), {"command": "rm -rf x"}).action == Action.PROMPT
    p.grant_prefix(["git", "pull"])
    assert p.evaluate(reg.get("bash"), {"command": "git pull origin main"}).action == Action.ALLOW and p.evaluate(reg.get("bash"), {"command": "git push"}).action == Action.PROMPT


def test_command_classes_and_prefix_rules_that_cannot_be_remembered():
    destructive = ["rm -rf build", "rm --recursive x", "sudo apt update", "mkfs.ext4 /dev/sda1", "dd if=/dev/zero of=x",
                   "git push --force", "git reset --hard HEAD~1", "git clean -fd", "docker system prune", "ls && rm -rf /tmp/x"]
    network = ["curl https://x", "ssh host ls", "git pull", "git clone x", "pip install x", "python3 -m pip install x",
               "uv pip install x", "npm i x", "cargo install x", "brew install x", "gh pr create"]
    interactive = ["vim a.py", "less log", "top", "python", "git rebase -i HEAD~3"]
    plain = ["ls -la", "pytest -q", "git status", "cat x | grep y", "python -c 'print(1)'", "rm x.tmp"]
    assert all("destructive" in classify_command(c).categories for c in destructive)
    assert all("network" in classify_command(c).categories for c in network)
    assert all("interactive" in classify_command(c).categories for c in interactive)
    assert all(classify_command(c).categories == [] for c in plain) and classify_command("git status; curl x").categories == ["network"]
    assert validate_prefix(["git", "pull"], "git pull origin main") == ""
    assert "too broad" in validate_prefix(["rm", "-rf"], "rm -rf x") and "too broad" in validate_prefix(["python3"], "python3 x.py")
    assert "single-token" in validate_prefix(["make"], "make test") and "heredoc" in validate_prefix(["cat", "-n"], "cat -n <<EOF\nx\nEOF")
    assert "match the start" in validate_prefix(["git", "push"], "git pull")
