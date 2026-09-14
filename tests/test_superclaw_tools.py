import io
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

from supergraph import SuperGraph

from superclaw import checks, review
from superclaw.attach import read as read_attachments
from superclaw.clipboard import parse_drop
from superclaw.delegate import SPAWN_KEY
from superclaw.observations import ObservationStore, ref_in
from superclaw.sandbox import Bubblewrap, Grant, detect
from superclaw.settings import LIMITS, Settings
from superclaw.tools import PathEscapes, Permission, Registry, Result, Safety, SideEffect, Tool, ToolContext, download, fetch, files, jail, relative, web
from superclaw.tools.budget import Category
from superclaw.tools.download import Download
from superclaw.tools.fetch import WebFetch
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


class FakeResponse:
    def __init__(self, url, body, content_type="text/html; charset=utf-8", status=200):
        self.url, self.body, self.headers, self.status, self.pos = url, body, {"Content-Type": content_type}, status, 0

    def read(self, n=-1):
        chunk = self.body[self.pos:] if n < 0 else self.body[self.pos:self.pos + n]
        self.pos += len(chunk)
        return chunk

    def geturl(self):
        return self.url

    def __enter__(self):
        self.pos = 0
        return self

    def __exit__(self, *exc):
        return False


PAGE_HTML = (
    '<!doctype html><html><head><title>T</title><style>b{}</style></head><body><nav><a href="/docs">Docs</a></nav>'
    '<h1>Hello <b>world</b></h1><p>First para with <a href="https://x.y/z">a link</a> and <a href="#top">anchor</a>.</p>'
    '<ul><li>one</li><li>two</li></ul><pre>code  here</pre><script>alert(1)</script></body></html>'
)


def test_jail_and_boundary(tmp_path, ws, tmp_path_factory, monkeypatch):
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
    pages = {"https://public.example/page": FakeResponse("https://public.example/page", PAGE_HTML.encode()),
             "https://public.example/data": FakeResponse("https://public.example/data", b'{"a": 1}', "application/json"),
             "https://public.example/latin": FakeResponse("https://public.example/latin", b"caf\xe9", "text/plain; charset=latin-1"),
             "https://public.example/long": FakeResponse("https://public.example/long", b"abcdefgh", "text/plain")}

    def open_url(request):
        if request.full_url.endswith("/missing"):
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, io.BytesIO(b"nope"))
        return pages[request.full_url]

    hosts = {"public.example": ["93.184.216.34"], "inner.example": ["10.0.0.5"], "both.example": ["93.184.216.34", "127.0.0.1"], "v6.example": ["::ffff:192.168.1.2"]}
    monkeypatch.setattr(fetch, "resolve", lambda host: hosts[host])
    monkeypatch.setattr(fetch, "open_url", open_url)
    monkeypatch.setattr(fetch, "browser_client", lambda: None)
    web_fetch = WebFetch(store)
    summary = web_fetch.run({"url": "https://public.example/page"}, ctx)
    assert summary.ok and summary.output.startswith("URL: https://public.example/page\nStatus: 200\nContent-Type: text/html; charset=utf-8\nBytes: ") and "Converted: html to markdown" in summary.output
    ref = ref_in(summary.output)
    assert f"Stored: §{ref} for {LIMITS.web_raw_ttl_days} days" in summary.output and "Title: Hello world" in summary.output and "Outline:\n# Hello world" in summary.output
    assert "First para" not in summary.output and "First para with [a link](https://x.y/z)" in store.load(ref).body and "full" not in summary.meta
    unread = web_fetch.run({"url": "https://public.example/page", "prompt": "what is it"}, ctx)
    assert "No child agent" in unread.output and "First para" not in unread.output
    ctx.state[SPAWN_KEY] = lambda a: Result.success(f"child read {a['refs'][0]} for: {a['task'].splitlines()[0]}")
    answered = web_fetch.run({"url": "https://public.example/page", "prompt": "what is it"}, ctx)
    assert answered.ok and answered.output.endswith(f"\n\nchild read {ref} for: what is it") and f"Stored: §{ref}" in answered.output
    page = web_fetch.run({"url": "https://public.example/page", "inline": True}, ctx)
    assert page.output.split("\n\n", 1)[1] == "[Docs](/docs)\n\n# Hello world\n\nFirst para with [a link](https://x.y/z) and anchor.\n\n- one\n- two\n\n```\ncode  here\n```"
    assert page.meta["full"] == page.output and "<h1>" in web_fetch.run({"url": "https://public.example/page", "format": "raw", "inline": True}, ctx).output
    data = web_fetch.run({"url": "https://public.example/data", "inline": True}, ctx).output
    assert data.endswith('\n\n{"a": 1}') and "Converted" not in data and web_fetch.run({"url": "https://public.example/latin", "inline": True}, ctx).output.endswith("\n\ncafé")
    cut = web_fetch.run({"url": "https://public.example/long", "max_bytes": 4, "inline": True}, ctx)
    assert cut.truncated and "Bytes: 4, truncated" in cut.output and cut.output.endswith("\n\nabcd")
    missing = web_fetch.run({"url": "https://public.example/missing"}, ctx)
    assert not missing.ok and missing.output == "Error fetching URL: HTTP 404 Not Found\nnope"
    for bad, why in (("http://inner.example/", "private"), ("http://both.example/", "loopback"), ("http://localhost:8000/", "localhost"), ("http://127.0.0.1/", "loopback"),
                     ("http://[::1]/", "loopback"), ("http://v6.example/", "private"), ("http://169.254.169.254/latest", "link-local"), ("ftp://public.example/x", "only public http"),
                     ("file:///etc/passwd", "only public http")):
        out = web_fetch.run({"url": bad}, ctx).output
        assert out.startswith("Error:") and why in out, (bad, out)
    with pytest.raises(fetch.Unsafe):
        fetch._Redirects().redirect_request(urllib.request.Request("https://public.example/page"), None, 302, "Found", {}, "http://inner.example/")
    assert WebFetch.deferred and WebFetch.safety.side_effect is SideEffect.NETWORK and fetch._Redirects.max_redirections == LIMITS.web_fetch_redirects

    class FakeBrowserResponse:
        def __init__(self, status, headers, content):
            self.status_code, self.headers, self.content = status, headers, content

    class FakeBrowser:
        calls: list[str] = []
        table = {"https://public.example/hop": FakeBrowserResponse(302, {"Location": "/landing"}, b""),
                 "https://public.example/landing": FakeBrowserResponse(200, {"Content-Type": "text/plain"}, b"landed"),
                 "https://public.example/leak": FakeBrowserResponse(302, {"Location": "http://inner.example/"}, b""),
                 "https://public.example/blocked": FakeBrowserResponse(403, {"Content-Type": "text/html"}, b"<html>nope</html>"),
                 "https://reader.example/https://public.example/blocked": FakeBrowserResponse(200, {"Content-Type": "text/plain"}, b"Title: X\n\nMarkdown Content:\nreader text")}

        def get(self, url):
            self.calls.append(url)
            return self.table[url]

    hosts["reader.example"] = ["93.184.216.34"]
    monkeypatch.setattr(fetch, "browser_client", lambda: FakeBrowser())
    landed = web_fetch.run({"url": "https://public.example/hop", "inline": True}, ctx)
    assert landed.output.startswith("URL: https://public.example/landing\nStatus: 200\n") and landed.output.endswith("\n\nlanded") and FakeBrowser.calls == ["https://public.example/hop", "https://public.example/landing"]
    assert "private" in web_fetch.run({"url": "https://public.example/leak"}, ctx).output
    blocked = web_fetch.run({"url": "https://public.example/blocked"}, ctx)
    assert not blocked.ok and blocked.output.startswith("Error fetching URL: HTTP 403 Forbidden; a reader proxy in SUPERCLAW_READER") and "nope" in blocked.output
    proxied = WebFetch(store, Settings.from_env({"SUPERCLAW_READER": "https://reader.example/"})).run({"url": "https://public.example/blocked", "inline": True}, ctx)
    assert proxied.ok and "Via: https://reader.example/" in proxied.output and proxied.output.endswith("\nreader text")
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
    monkeypatch.setattr(web, "ddgs_search", lambda query, limit: None)
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
    monkeypatch.setattr(web, "ddgs_search", lambda query, limit: [web.Hit("Engine", "https://engine.example/", "served"), web.Hit("More", "https://more.example/", "")][:limit])
    assert duck.run({"query": "any", "limit": 1}, ctx).output == "Results from duckduckgo for: any\n1. Engine\n   https://engine.example/\n   served"

    def refuse(query, limit):
        raise web.Refused("the search engines refused the query")

    monkeypatch.setattr(web, "ddgs_search", refuse)
    assert duck.run({"query": "any"}, ctx).output.startswith("Error: the search engines refused")
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
    assert reg.run("grep", {"pattern": "beta", "case_insensitive": True, "output_mode": "count"}, ctx).output.splitlines() == ["src/a.py:1", "src/b.txt:2"]
    assert reg.run("grep", {"pattern": "beta", "output_mode": "files_with_matches", "path": "src"}, ctx).output == "src/b.txt"
    assert reg.run("grep", {"pattern": "beta", "glob": "*.txt", "head_limit": 1, "case_insensitive": True}, ctx).output.splitlines() == ["src/b.txt:1:Beta here", "[... more matches; raise head_limit or narrow the pattern ...]"]
    assert reg.run("grep", {"pattern": "(?<=B)ETA"}, ctx).output == "src/a.py:2:BETA" and reg.run("grep", {"pattern": "beta", "path": "src/a.py", "case_insensitive": True}, ctx).output == "src/a.py:2:BETA"
    with monkeypatch.context() as plain:
        plain.setattr(files, "which", lambda name: None)
        assert reg.run("grep", {"pattern": "beta", "case_insensitive": True}, ctx).output.splitlines() == ["src/a.py:2:BETA", "src/b.txt:1:Beta here", "src/b.txt:2:and beta again"]
        assert reg.run("grep", {"pattern": "beta", "case_insensitive": True, "output_mode": "count"}, ctx).output.splitlines() == ["src/a.py:1", "src/b.txt:2"]
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


def test_bash(tmp_path, monkeypatch):
    ctx = ToolContext(workspace=tmp_path)
    (tmp_path / "sub").mkdir()
    bash = Bash()
    assert bash.run({"command": "echo hi"}, ctx).output == "hi"
    assert "[exit 3]" in bash.run({"command": "echo bad >&2; exit 3"}, ctx).output
    assert bash.run({"command": "pwd", "cwd": "sub"}, ctx).output == str((tmp_path / "sub").resolve())
    assert "timed out" in bash.run({"command": "sleep 5", "timeout_ms": 200}, ctx).output
    assert bash.category({"command": "pytest -q"}) is Category.TEST and bash.category({"command": "ls"}) is Category.PROCESS
    fake = tmp_path / "bin" / "yt-dlp"
    fake.parent.mkdir()
    fake.write_text('#!/bin/sh\next=mp4\nwhile [ $# -gt 1 ]; do case "$1" in --paths) dest="$2"; shift;; --extract-audio) ext=mp3;; esac; shift; done\n'
                    'case "$1" in *unsupported*) echo "ERROR: Unsupported URL: $1" >&2; exit 1;; esac\nf="$dest/clip-abc.$ext"\nprintf data > "$f"\necho "$f"\n')
    fake.chmod(0o755)
    monkeypatch.setattr(download, "which", lambda name: str(fake))
    monkeypatch.setattr(fetch, "resolve", lambda host: ["93.184.216.34"])
    files = {"https://public.example/unsupported/report.pdf": FakeResponse("https://public.example/unsupported/report.pdf", b"%PDF", "application/pdf"),
             "https://public.example/x": FakeResponse("https://public.example/x", b"notes", "text/plain")}
    files["https://public.example/x"].headers["Content-Disposition"] = 'attachment; filename="served.bin"'
    monkeypatch.setattr(download, "open_url", lambda request: files[request.full_url])
    dl = Download()
    res = dl.run({"url": "https://public.example/watch?v=1"}, ctx)
    assert res.ok and res.output == "Saved clip-abc.mp4 (4B)" and res.changed_files == ["clip-abc.mp4"] and (tmp_path / "clip-abc.mp4").read_bytes() == b"data"
    assert dl.run({"url": "https://public.example/watch?v=1", "kind": "audio", "dest": "media"}, ctx).output == "Saved media/clip-abc.mp3 (4B)"
    assert dl.run({"url": "https://public.example/unsupported/report.pdf"}, ctx).output == "Saved report.pdf (4B)" and (tmp_path / "report.pdf").read_bytes() == b"%PDF"
    assert dl.run({"url": "https://public.example/x", "kind": "file"}, ctx).output == "Saved served.bin (5B)"
    assert dl.run({"url": "https://public.example/x", "kind": "file", "name": "../../notes.txt"}, ctx).output == "Saved notes.txt (5B)"
    exists = dl.run({"url": "https://public.example/x", "kind": "file"}, ctx)
    assert not exists.ok and "already exists" in exists.output and "escapes" in dl.run({"url": "https://public.example/x", "dest": "../out"}, ctx).output
    assert "Unsupported URL" in dl.run({"url": "https://public.example/unsupported/clip", "kind": "video"}, ctx).output
    monkeypatch.setattr(download, "which", lambda name: None)
    assert "yt-dlp is not installed" in dl.run({"url": "https://public.example/watch", "kind": "video"}, ctx).output
    monkeypatch.setattr(fetch, "resolve", lambda host: ["10.0.0.9"])
    assert "private" in dl.run({"url": "https://public.example/x", "kind": "file"}, ctx).output
    assert Download.deferred and Download.safety.side_effect is SideEffect.NETWORK
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
