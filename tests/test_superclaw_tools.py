import json
import os
import subprocess
import sys

import pytest

from supergraph import SuperGraph

from superclaw import checks, review
from superclaw.attach import read as read_attachments
from superclaw.clipboard import parse_drop
from superclaw.observations import ObservationStore
from superclaw.sandbox import Bubblewrap, Grant, detect
from superclaw.settings import LIMITS, Settings
from superclaw.tools import PathEscapes, Permission, Registry, Result, Safety, SideEffect, Tool, ToolContext, jail, relative, web
from superclaw.tools.budget import Category
from superclaw.tools.files import core_file_tools
from superclaw.tools.shell import Bash
from superclaw.tools.web import WebSearch
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


DUCK_HTML = (
    '<div class="result web-result"><h2 class="result__title"><a rel="nofollow" class="result__a" href="https://textual.textualize.io/">Textual</a></h2>'
    '<a class="result__snippet" href="https://textual.textualize.io/"><b>Textual</b> is a <b>TUI</b> framework for <b>Python</b>.</a></div>'
    '<div class="result result--ad"><h2 class="result__title"><a rel="nofollow" class="result__a" href="https://duckduckgo.com/y.js?ad_provider=x">Buy TUI</a></h2>'
    '<a class="result__snippet" href="https://duckduckgo.com/y.js?ad_provider=x">ad copy</a></div>'
    '<div class="result"><h2 class="result__title"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Frealpython.com%2Fpython-textual%2F&amp;rut=1">Python Textual: Build UIs</a></h2>'
    '<a class="result__snippet" href="#">Learn &amp; build.</a></div>'
)


def test_file_tools(reg, ws, monkeypatch):
    ctx = ToolContext(workspace=ws)
    calls, pauses = [], []

    def fake_fetch(url, headers, data=None):
        calls.append((url, data))
        if url.startswith("https://www.googleapis.com/"):
            return json.dumps({"items": [{"title": "Textual", "link": "https://textual.textualize.io/", "snippet": "TUI framework"}]}).encode()
        if data and (b"q=blocked" in data or (b"q=flaky" in data and sum(b"q=flaky" in (d or b"") for _, d in calls) == 1)):
            return b"<html><div id='anomaly-modal'>challenge</div></html>"
        if data and b"q=nothing" in data:
            return b"<html><div class='no-results'></div></html>"
        return DUCK_HTML.encode()

    monkeypatch.setattr(web, "fetch", fake_fetch)
    monkeypatch.setattr(web, "pause", pauses.append)
    monkeypatch.setattr(web, "now", lambda: 0.0)
    duck = WebSearch(Settings.from_env({}))
    out = duck.run({"query": "textual tui"}, ctx).output
    assert out == ("Results from duckduckgo for: textual tui\n1. Textual\n   https://textual.textualize.io/\n   Textual is a TUI framework for Python.\n"
                   "2. Python Textual: Build UIs\n   https://realpython.com/python-textual/\n   Learn & build.")
    assert calls[-1][0] == "https://html.duckduckgo.com/html/" and b"q=textual+tui" in calls[-1][1]
    scoped = duck.run({"query": "x", "domains": ["realpython.com"]}, ctx).output
    assert "1. Python Textual" in scoped and "textualize" not in scoped and duck.run({"query": "x", "limit": 1}, ctx).output.count("\n   https://") == 1
    assert duck.run({"query": "nothing"}, ctx).output == "No results for: nothing" and duck.run({"query": " "}, ctx).output.startswith("Error:")
    refused = duck.run({"query": "blocked"}, ctx)
    assert not refused.ok and "DuckDuckGo refused" in refused.output and "GOOGLE_API_KEY" in refused.output
    assert sum(b"q=blocked" in (d or b"") for _, d in calls) == LIMITS.web_search_attempts and LIMITS.web_search_retry_s in pauses
    assert duck.run({"query": "flaky"}, ctx).ok and sum(b"q=flaky" in (d or b"") for _, d in calls) == 2 and pauses.count(LIMITS.web_search_min_interval_s) >= 2
    keyed = Settings.from_env({"GOOGLE_API_KEY": "k", "GOOGLE_CSE_ID": "c"})
    assert keyed.search_engine == "google" and Settings.from_env({"SUPERCLAW_SEARCH": "duckduckgo", "GOOGLE_API_KEY": "k", "GOOGLE_CSE_ID": "c"}).search_engine == "duckduckgo"
    assert WebSearch(keyed).run({"query": "textual"}, ctx).output == "Results from google for: textual\n1. Textual\n   https://textual.textualize.io/\n   TUI framework"
    assert "key=k" in calls[-1][0] and "num=5" in calls[-1][0] and "GOOGLE_API_KEY" in WebSearch(Settings.from_env({"SUPERCLAW_SEARCH": "google"})).run({"query": "x"}, ctx).output
    assert WebSearch.deferred and WebSearch.safety.side_effect is SideEffect.NETWORK and WebSearch.output_category is Category.SEARCH
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
    spaced = ws / "my notes.txt"
    spaced.write_text("n")
    assert parse_drop(f"'{spaced}'") == [spaced] and parse_drop(f"file://{ws}/my%20notes.txt {ws}/shot.png") == [spaced, ws / "shot.png"]
    assert parse_drop(f"{spaced} {ws}/missing.txt") == [] and parse_drop("just some words") == [] and parse_drop("relative/path.txt") == []
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
    with pytest.raises(review.ReviewError):
        review.prompt(review.uncommitted(repo))
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / "notes.txt").write_text("todo\n")
    subprocess.run(["git", "add", "calc.py"], cwd=repo, check=True, capture_output=True)
    dirty = review.uncommitted(repo)
    assert "+    return a - b" in dirty.patch and dirty.untracked == ["notes.txt"]
    text = review.prompt(dirty, "focus on arithmetic")
    assert text.startswith("Review the change") and "<focus>\nfocus on arithmetic\n</focus>" in text and "notes.txt" in text and 'scope="uncommitted changes against HEAD"' in text
    subprocess.run("git -c user.email=t@t -c user.name=t commit -q -m add-calc".split(), cwd=repo, check=True, capture_output=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert "+    return a - b" in review.commit(repo, head).patch and review.commit(repo, head).label.endswith("add-calc")
    subprocess.run(["git", "checkout", "-q", "-b", "feature"], cwd=repo, check=True, capture_output=True)
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    subprocess.run("git -c user.email=t@t -c user.name=t commit -q -am fix".split(), cwd=repo, check=True, capture_output=True)
    assert "+    return a + b" in review.against(repo, "main").patch and review.against(repo, "feature").patch == ""
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='x'\n")
    (project / "go.mod").write_text("module x\n")
    (project / "Cargo.toml").write_text("[package]\n")
    (project / "pnpm-lock.yaml").write_text("")
    (project / "package.json").write_text(json.dumps({"scripts": {"test": "vitest", "lint": "eslint .", "typecheck": "tsc"}}))
    found = checks.detect(project)
    assert [c.id for c in found] == ["go.test", "pnpm.typecheck", "pnpm.test", "pnpm.lint", "python.pytest", "cargo.test"]
    assert checks.detect(tmp_path / "plain") == [] and found[1].command == ["pnpm", "run", "typecheck"]
    fake = [checks.Check("a.pass", "A", [sys.executable, "-c", "print('fine')"], "test"),
            checks.Check("b.fail", "B", [sys.executable, "-c", "import sys; print('boom line'); sys.exit(3)"], "test"),
            checks.Check("c.slow", "C", [sys.executable, "-c", "import time; time.sleep(5)"], "test"),
            checks.Check("d.missing", "D", ["no-such-binary-xyz"], "test")]
    report = checks.run(project, fake, timeout_s=1)
    assert [r.status for r in report.results] == ["passed", "failed", "timed_out", "error"] and not report.ok and report.results[1].exit_code == 3
    assert report.results[1].tail == ["boom line"] and checks.run(project, fake, only=("a.pass",)).ok
    assert checks.lines(report)[0].startswith("[pass] A:") and "    boom line" in checks.lines(report) and checks.as_json(report)["results"][2]["status"] == "timed_out"
    assert 'check id="b.fail"' in checks.remediation_prompt(report) and "do not weaken" in checks.remediation_prompt(report)
