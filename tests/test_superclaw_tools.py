import os
import subprocess

import pytest

from supergraph import SuperGraph

from superclaw.attach import read as read_attachments
from superclaw.observations import ObservationStore
from superclaw.sandbox import Bubblewrap, Grant, detect
from superclaw.settings import LIMITS
from superclaw.tools import PathEscapes, Permission, Registry, Result, Safety, SideEffect, Tool, ToolContext, jail, relative
from superclaw.tools.budget import Category
from superclaw.tools.files import core_file_tools
from superclaw.tools.shell import Bash
from superclaw.worktree import WorktreeError, prepare


class Leaky(Tool):
    name = "leaky"
    description = "Return many secrets."
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}
    safety = Safety(SideEffect.NONE, Permission.ALLOW, "No side effects.")

    def run(self, args, ctx):
        if args.get("boom"):
            raise RuntimeError("kaboom")
        return Result.success("\n".join(f"secret_token=abc{i}" for i in range(20_000)))


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("alpha\nbeta\ngamma\n")
    (tmp_path / "src" / "b.txt").write_text("Beta here\nand beta again\n")
    return tmp_path


@pytest.fixture
def reg():
    reg = Registry()
    for tool in core_file_tools():
        reg.register(tool)
    return reg


def test_jail_and_boundary(tmp_path, ws, tmp_path_factory):
    other = tmp_path_factory.mktemp("other")
    (other / "secret").write_text("s")
    os.symlink(other / "secret", ws / "link")
    assert jail(ws, "src/a.py") == (ws / "src" / "a.py").resolve()
    for escape in ("../outside.txt", str(other / "x"), "link"):
        with pytest.raises(PathEscapes):
            jail(ws, escape)
    roots = (ws, other)
    assert jail(roots, str(other / "x")) == (other / "x").resolve() and jail(roots, "src/a.py") == (ws / "src" / "a.py").resolve()
    assert relative(roots, (other / "secret").resolve()) == "secret" and relative(ws, (other / "secret").resolve()) == str((other / "secret").resolve())
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    store = ObservationStore(gs)
    reg = Registry(observations=store)
    reg.register(Leaky())
    ctx = ToolContext(workspace=tmp_path, session_id="s1")
    res = reg.run("leaky", {}, ctx, call_id="call_9")
    saved = store.load(res.artifact.ref)
    assert "abc1\n" not in saved.body and "[REDACTED]" in saved.body and saved.body.count("\n") == 19_999
    assert res.diagnostics.model_tokens <= LIMITS.tool_output_tokens and f"recall §{res.artifact.ref} for the full output" in res.output
    assert reg.run("leaky", {}, ctx, call_id="call_9").artifact.ref == res.artifact.ref and store.save("s1", "leaky", "call_9", "other") != res.artifact.ref
    assert "kaboom" in reg.run("leaky", {"boom": True}, ctx).output and "unknown tool" in reg.run("nope", {}, ctx).output
    bare = Registry()
    bare.register(Leaky())
    assert "not recoverable" in bare.run("leaky", {}, ctx).output
    gs.close()


def test_file_tools(reg, ws):
    ctx = ToolContext(workspace=ws)
    assert reg.run("read_file", {"path": "src/a.py"}, ctx).output == "1→alpha\n2→beta\n3→gamma"
    assert "already in your context" in reg.run("read_file", {"path": "src/a.py", "offset": 2, "limit": 1}, ctx).output
    assert reg.run("read_file", {"path": "src/a.py", "offset": 2, "limit": 1, "force": True}, ctx).output == "2→beta\n[1 more lines; call read_file with offset=3 to continue]"
    (ws / "big.txt").write_text("\n".join(["x" * 3000] + [f"l{i}" for i in range(LIMITS.read_file_lines + 5)]))
    res = reg.run("read_file", {"path": "big.txt"}, ctx)
    assert res.output.count("\n") == LIMITS.read_file_lines and "… [+1,000 chars]" in res.output.split("\n")[0]
    edit = {"path": "src/a.py", "description": "d", "old_string": "beta", "new_string": "BETA"}
    (ws / "src" / "a.py").write_text("alpha\nbeta\ngamma\ndelta\n")
    assert "changed on disk" in reg.run("edit_file", edit, ctx).output
    reg.run("read_file", {"path": "src/a.py"}, ctx)
    res = reg.run("edit_file", edit, ctx)
    assert res.output == "Replaced 1 occurrence(s) in src/a.py" and res.display.kind == "diff" and "-beta\n+BETA" in res.display.preview
    assert "5 times" in reg.run("edit_file", {**edit, "old_string": "a", "new_string": ""}, ctx).output
    assert "read the file before" in reg.run("write_file", {"path": "src/b.txt", "description": "d", "content": "x", "overwrite": True}, ctx).output
    assert reg.run("write_file", {"path": "new/f.txt", "description": "d", "content": "x\n"}, ctx).changed_files == ["new/f.txt"]
    (ws / "bin.dat").write_bytes(b"beta\0")
    (ws / "huge.txt").write_text("beta\n" + "x" * (LIMITS.read_file_bytes + 1))
    assert reg.run("grep", {"pattern": "beta", "case_insensitive": True}, ctx).output.splitlines() == ["src/a.py:2:BETA", "src/b.txt:1:Beta here", "src/b.txt:2:and beta again"]
    assert reg.run("glob", {"pattern": "**/*.py"}, ctx).output.splitlines() == ["src/a.py"]
    assert reg.run("list_directory", {"path": "src"}, ctx).output.splitlines() == ["a.py", "b.txt"]
    (ws / "shot.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 10)
    attached = read_attachments(["src/a.py", "shot.png", "missing.txt", "../escape"], (ws,))
    assert '<attachment path="src/a.py">' in attached.text and "alpha" in attached.text
    assert attached.images[0].startswith("data:image/png;base64,") and '<attachment path="shot.png">\n(attached as an image)' in attached.text
    assert len(attached.problems) == 2 and any("not a file" in p for p in attached.problems) and any("escapes" in p for p in attached.problems)
    vendor = ws.parent / "vendor"
    vendor.mkdir()
    (vendor / "lib.py").write_text("vendored\n")
    assert "escapes the workspace" in reg.run("read_file", {"path": str(vendor / "lib.py")}, ctx).output
    wide = ToolContext(workspace=ws, extra_dirs=(vendor,))
    assert wide.roots == (ws, vendor) and reg.run("read_file", {"path": str(vendor / "lib.py")}, wide).output == "1→vendored"
    assert reg.run("grep", {"pattern": "vendored", "path": str(vendor)}, wide).output == "lib.py:1:vendored"


def test_bash(tmp_path):
    ctx = ToolContext(workspace=tmp_path)
    (tmp_path / "sub").mkdir()
    bash = Bash()
    assert bash.run({"command": "echo hi"}, ctx).output == "hi"
    assert "[exit 3]" in bash.run({"command": "echo bad >&2; exit 3"}, ctx).output
    assert bash.run({"command": "pwd", "cwd": "sub"}, ctx).output == str((tmp_path / "sub").resolve())
    assert "timed out" in bash.run({"command": "sleep 5", "timeout_ms": 200}, ctx).output
    assert bash.category({"command": "pytest -q"}) is Category.TEST and bash.category({"command": "ls"}) is Category.PROCESS
    argv = Bubblewrap().wrap(["bash", "-c", "x"], tmp_path, tmp_path, Grant(network=True, paths=["/opt/extra"]))
    assert argv[0] == "bwrap" and "--unshare-net" not in argv and "/opt/extra" in argv
    if detect():
        tool = Bash(detect())
        assert tool.run({"command": "echo in > made.txt && cat made.txt"}, ctx).output == "in"
        assert not tool.run({"command": "touch /etc/superclaw-probe"}, ctx).ok
        assert not tool.run({"command": "curl -sm2 https://example.com"}, ctx).ok
        ctx.state["approval"] = {"escalated": True, "network": False}
        assert tool.run({"command": "test -w /tmp && echo host"}, ctx).output == "host"
    repo, trees, plain = tmp_path / "repo", tmp_path / "trees", tmp_path / "plain"
    repo.mkdir()
    plain.mkdir()
    for cmd in ("git init -q -b main", "git -c user.email=t@t -c user.name=t commit -q --allow-empty -m init"):
        subprocess.run(cmd.split(), cwd=repo, check=True, capture_output=True)
    first = prepare(repo, trees, "alpha")
    assert first.path.is_dir() and first.branch == "superclaw/alpha" and not first.reused and first.repo_root == repo.resolve()
    again = prepare(repo, trees, "alpha")
    assert again.reused and again.path == first.path and (first.path / ".git").exists()
    assert prepare(first.path, trees, "beta").path.parent == first.path.parent
    for bad_cwd, bad_name in ((repo, "bad/name"), (repo, ""), (plain, "gamma")):
        with pytest.raises(WorktreeError):
            prepare(bad_cwd, trees, bad_name or "x" * (LIMITS.worktree_name_chars + 1))
