import json
import shutil
import subprocess
from pathlib import Path

import pytest

from superclaw.agents import load_agents
from superclaw.hooks import Dispatcher, load_hooks
from superclaw.mcp import load_config
from superclaw.plugins import PluginError, add_marketplace, claude_installed, install_from_marketplace, is_marketplace_ref, load_marketplaces, load_plugins, remove_marketplace, resolve_source
from superclaw.plugins import install as install_plugin
from superclaw.plugins import remove as remove_plugin
from superclaw.agents import resolve as resolve_agent
from superclaw.policy import Mode
from superclaw import tooling
from superclaw.prompt import PromptInputs, build_system_prompt, core_prompt, project_guidelines, skills_block
from superclaw.repomap import render as render_repo
from superclaw.repomap import scan as scan_repo
from superclaw.repomap import search as search_repo
from superclaw.runtime import approx_tokens
from superclaw.settings import LIMITS, Settings
from superclaw.skills import Skill, load_skills
from superclaw.tools import ToolContext
from superclaw.tools.skill import SkillTool
from superclaw.usercommands import expand, load_commands
from superclaw.usercommands import find as find_command


def test_prompt_assembly_guidelines_and_skills(tmp_path, monkeypatch):
    root = tmp_path
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    assert 0 < approx_tokens(core_prompt()) < 1000 and approx_tokens(build_system_prompt(PromptInputs(cwd=Path("/nonexistent"), mode=Mode.ASK))) < 1600
    (root / "AGENTS.md").write_text("ROOT RULES")
    (root / "SUPERCLAW.md").write_text("BRAND")
    svc = root / "services" / "api"
    svc.mkdir(parents=True)
    (svc / "agents.md").write_text("API RULES")
    out = project_guidelines(svc, root)
    assert out.index("ROOT RULES") < out.index("API RULES") and "BRAND" not in out and "## Project guidelines (services/api/agents.md)" in out
    (root / "AGENTS.md").write_text("x" * (LIMITS.guideline_file_bytes + 100))
    assert project_guidelines(root, root).count("x") <= LIMITS.guideline_file_bytes
    skill = tmp_path / "skills" / "bench"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: bench\ndescription: Run benchmarks.\n---\nBODY")
    skills = load_skills([tmp_path / "skills", tmp_path / "missing"])
    assert skills == [Skill("bench", "Run benchmarks.", "BODY", str(skill / "SKILL.md"))]
    block = skills_block([*skills, *(Skill(f"s{i:03d}", "d" * LIMITS.skill_description_chars, "", "p") for i in range(60))])
    assert "- bench: Run benchmarks." in block and "BODY" not in block and "more (call skill with a name" in block
    userfile = tmp_path / "SUPERCLAW.md"
    userfile.write_text("PERSONAL")
    prompt = build_system_prompt(PromptInputs(cwd=root, mode=Mode.PLAN, memory="- user prefers tabs", model="m", provider="p", skills=skills, user_guidelines=userfile))
    assert "Plan mode is active" in prompt and "user prefers tabs" in prompt and "Git branch: main" in prompt and prompt.index("PERSONAL") < prompt.index("xxxx")
    profiles = tmp_path / "agents"
    profiles.mkdir()
    (profiles / "reviewer.md").write_text("---\nname: reviewer\ndescription: Reviews diffs.\ntools: read_file, grep\nmodel: p/m\n---\nOnly review.")
    (profiles / "notes.txt").write_text("ignored")
    loaded = load_agents([profiles, tmp_path / "missing"])
    assert [a.name for a in loaded] == ["reviewer"] and loaded[0].tools == frozenset({"read_file", "grep"}) and loaded[0].model == "p/m"
    assert resolve_agent("reviewer", [profiles]).prompt == "Only review."
    builtin = {a.name: a for a in load_agents(Settings.from_env({"XDG_CONFIG_HOME": str(tmp_path / "empty")}).agent_roots())}
    assert set(builtin) == {"explore", "review"} and "write_file" not in builtin["explore"].tools and "bash" not in builtin["explore"].tools
    assert {"read_file", "grep", "glob"} <= builtin["explore"].tools and "findings, not files" in builtin["explore"].prompt
    assert "bash" in builtin["review"].tools and "write" not in builtin["review"].tools and builtin["review"].description.startswith("Read-only review")
    local = tmp_path / "cfgown" / "superclaw" / "agents"
    local.mkdir(parents=True)
    (local / "explore.md").write_text("---\nname: explore\ndescription: Mine.\n---\nMINE")
    assert resolve_agent("explore", Settings.from_env({"XDG_CONFIG_HOME": str(tmp_path / "cfgown")}).agent_roots()).prompt == "MINE"
    with pytest.raises(KeyError):
        resolve_agent("nope", [profiles])
    with_agent = build_system_prompt(PromptInputs(cwd=root, mode=Mode.ASK, model="m", provider="p", agent=loaded[0].prompt))
    assert "<agent>" in with_agent and "Only review." in with_agent and "never widens" in with_agent
    assert "<agent>" not in build_system_prompt(PromptInputs(cwd=root, mode=Mode.ASK, model="m", provider="p"))
    equipped = build_system_prompt(PromptInputs(cwd=root, mode=Mode.ASK, model="m", provider="p", tools=("rg", "fd", "jq")))
    boxed = build_system_prompt(PromptInputs(cwd=root, mode=Mode.ASK, model="m", provider="p", sandbox="bubblewrap", extra_dirs=(root / "vendor",)))
    assert "bash runs in a bubblewrap sandbox" in boxed and "fresh /tmp and /dev/shm" in boxed and "and the additional directories" in boxed and "chmod 0700" in boxed
    assert "sandbox" not in equipped.split("<environment>")[1].split("</environment>")[0]
    assert "Host tools present: rg, fd, jq. In bash prefer rg over grep, fd over find." in equipped and "Host tools" not in build_system_prompt(PromptInputs(cwd=root, mode=Mode.ASK))
    assert tooling.guidance(("jq",)) == "Host tools present: jq." and tooling.guidance(()) == ""
    known = build_system_prompt(PromptInputs(cwd=root, mode=Mode.ASK, facts="- (today, https://x.example/) Fact one"))
    assert "<facts>\nFacts learned from sources" in known and "Fact one" in known and "<facts>\nFacts learned" not in equipped
    monkeypatch.setattr(tooling, "which", lambda name: "/usr/bin/" + name if name in ("rg", "uv") else None)
    assert tooling.host_tools() == ("rg", "uv")
    commands = tmp_path / "commands"
    commands.mkdir()
    (commands / "pr.md").write_text("---\ndescription: Open a PR.\nagent: reviewer\n---\nOpen a PR titled $1 for $ARGUMENTS; cost $$5")
    (commands / "Bad Name.md").write_text("ignored")
    (commands / "plain.md").write_text("Just do it.")
    loaded = load_commands([commands, tmp_path / "missing"])
    assert [c.name for c in loaded] == ["plain", "pr"] and loaded[1].agent == "reviewer" and loaded[0].description == "User command: /plain"
    assert expand(loaded[1].template, "42 fix the build") == "Open a PR titled 42 for 42 fix the build; cost $5"
    assert expand("$1 then $2 then $3", "a b") == "a then b then " and expand("Just do it.", "now") == "Just do it.\n\nnow" and expand("x", "") == "x"
    assert expand("title $1", '"fix build" now') == "title fix build" and expand("$1", "it's") == "it's"
    assert find_command("PR", [commands]).name == "pr" and find_command("nope", [commands]) is None
    (commands / "bench.md").write_text("File wins")
    with_skills = load_commands([commands], [tmp_path / "skills"])
    assert [c.name for c in with_skills] == ["bench", "plain", "pr"] and with_skills[0].template == "File wins" and not with_skills[0].skill
    (commands / "bench.md").unlink()
    as_skill = find_command("bench", [commands], [tmp_path / "skills"])
    assert as_skill.skill and as_skill.description == "Run benchmarks." and as_skill.path == str(skill / "SKILL.md") and find_command("bench", [commands]) is None
    assert expand(as_skill.template, "the parser") == '<skill name="bench">\nBODY\n</skill>\n\nFollow the skill above for this request. the parser'
    bundle = tmp_path / "bundle" / "nested"
    (bundle / "skills" / "deploy").mkdir(parents=True)
    (bundle / "agents").mkdir()
    (bundle / "commands").mkdir()
    (bundle / "plugin.json").write_text(json.dumps({"id": "acme-tools", "description": "Acme workflows.", "version": "1.2.0"}))
    (bundle / "skills" / "deploy" / "SKILL.md").write_text("---\nname: deploy\ndescription: Ship it.\n---\nSTEPS")
    (bundle / "agents" / "ops.md").write_text("---\nname: ops\ndescription: Ops.\n---\nRun ops.")
    (bundle / "commands" / "ship.md").write_text("Ship $ARGUMENTS")
    (bundle / "hooks.json").write_text("{}")
    settings = Settings.from_env({"XDG_CONFIG_HOME": str(tmp_path / "cfg")})
    installed = install_plugin(str(tmp_path / "bundle"), settings.user_plugins)
    assert installed.path == settings.user_plugins / "acme-tools" and installed.parts == ["skills", "agents", "commands", "hooks"] and installed.format == "superclaw"
    assert [p.id for p in load_plugins(settings.plugin_roots(root))] == ["acme-tools"] and settings.plugin_dirs(root) == [installed.path]
    assert any(s.name == "deploy" for s in load_skills(settings.skill_roots(root))) and any(a.name == "ops" for a in load_agents(settings.agent_roots(root)))
    assert find_command("ship", settings.command_roots(root)).template == "Ship $ARGUMENTS"
    with pytest.raises(PluginError):
        install_plugin(str(tmp_path / "bundle"), settings.user_plugins)
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "plugin.json").write_text(json.dumps({"id": "Bad Id"}))
    with pytest.raises(PluginError):
        install_plugin(str(tmp_path / "bad"), settings.user_plugins)
    assert remove_plugin("acme-tools", settings.user_plugins) == installed.path and load_plugins(settings.plugin_roots(root)) == []
    with pytest.raises(PluginError):
        remove_plugin("acme-tools", settings.user_plugins)
    claude = tmp_path / "hyper"
    (claude / ".claude-plugin").mkdir(parents=True)
    (claude / "skills" / "rulebook").mkdir(parents=True)
    (claude / "agents").mkdir()
    (claude / "hooks").mkdir()
    (claude / ".claude-plugin" / "plugin.json").write_text(json.dumps({"name": "Hyper", "description": "Skills and hooks.", "version": "1.5.0", "skills": "./skills/"}))
    (claude / "skills" / "rulebook" / "SKILL.md").write_text("---\nname: rulebook\ndescription: Laws.\ntriggers:\n  - \"build\"\n---\nLAWS")
    (claude / "agents" / "checks.md").write_text("# Checks\n\nPlain agent body.")
    (claude / "hooks" / "start.sh").write_text('#!/bin/sh\nprintf \'{"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "bootstrap from %s"}}\' "$CLAUDE_PLUGIN_ROOT"\n')
    (claude / "hooks" / "start.sh").chmod(0o755)
    (claude / "hooks" / "hooks.json").write_text(json.dumps({"hooks": {"SessionStart": [{"matcher": "startup|clear|compact", "hooks": [{"type": "command", "command": 'sh "${CLAUDE_PLUGIN_ROOT}/hooks/start.sh"', "timeout": 5}]}],
                                                                       "Notification": [{"hooks": [{"type": "command", "command": "true"}]}]}}))
    (claude / ".mcp.json").write_text(json.dumps({"mcpServers": {"hyper": {"command": "node", "args": ["${CLAUDE_PLUGIN_ROOT}/bin/server.mjs"]}}}))
    linked = install_plugin(str(claude), settings.user_plugins, link=True)
    assert linked.id == "hyper" and linked.format == "claude" and linked.path.is_symlink() and linked.parts == ["skills", "agents", "hooks", "mcp"]
    assert [p.id for p in settings.plugins(root)] == ["hyper"] and settings.plugin_dirs(root) == [linked.path]
    assert any(s.name == "hyper:rulebook" and s.content == "LAWS" for s in load_skills(settings.skill_roots(root))) and any(a.name == "checks" for a in load_agents(settings.agent_roots(root)))
    (settings.config_dir / "skills" / "acme" / "ship").mkdir(parents=True)
    (settings.config_dir / "skills" / "acme" / "ship" / "SKILL.md").write_text("---\nname: ship\ndescription: Ship.\n---\nSHIP")
    (settings.config_dir / "skills" / "linked").symlink_to(claude / "skills", target_is_directory=True)
    names = {s.name for s in load_skills(settings.skill_roots(root))}
    assert {"acme:ship", "linked:rulebook", "hyper:rulebook"} <= names
    skill_tool = SkillTool(settings.skill_roots(root))
    assert skill_tool.run({"name": "hyper:rulebook"}, ToolContext(workspace=root)).output == "LAWS" and skill_tool.run({"name": "ship"}, ToolContext(workspace=root)).output == "SHIP"
    assert "unknown skill 'rulebook'" in skill_tool.run({"name": "rulebook"}, ToolContext(workspace=root)).output
    hooks = load_hooks([(linked.hooks, linked.path)])
    assert [h.event for h in hooks] == ["sessionStart", "notification"] and hooks[0].command[:2] == ["/bin/sh", "-c"] and str(linked.path) in hooks[0].command[2] and hooks[0].timeout_s == 5
    dispatcher = Dispatcher(hooks, root)
    assert dispatcher.dispatch("sessionStart", {"session": "s1", "prompt": "hi"}, "startup").context == [f"bootstrap from {linked.path}"]
    assert dispatcher.dispatch("sessionStart", {"session": "s1", "prompt": "hi"}, "resume").context == []
    servers = load_config([linked.mcp]).servers
    assert servers[0].name == "hyper" and servers[0].args == [f"{linked.path}/bin/server.mjs"]
    assert remove_plugin("hyper", settings.user_plugins) == linked.path and claude.is_dir() and settings.plugins(root) == []
    home = tmp_path / "home" / ".claude"
    (home / "plugins").mkdir(parents=True)
    (home / "plugins" / "installed_plugins.json").write_text(json.dumps({"version": 2, "plugins": {"hyper@hyper": [{"scope": "user", "installPath": str(claude)}], "gone@x": [{"installPath": str(tmp_path / "missing")}]}}))
    assert claude_installed(home) == [claude] and claude_installed(tmp_path / "nohome") == []
    assert settings.claude_config is False and settings.claude_settings(root) == [] and settings.claude_roots("skills", root) == []
    for rel, text in (("skills/tidy/SKILL.md", "---\nname: tidy\ndescription: |\n  Tidy.\n  Up.\nlicense: MIT\n---\nTIDY"), ("agents/reviewer.md", "---\nname: reviewer\ndescription: Reviews.\n---\nREVIEW"),
                      ("commands/plan.md", "Plan $ARGUMENTS"), (".claude.json", json.dumps({"mcpServers": {"docs": {"command": "docs-mcp"}, "off": {"command": "x"}}, "disabledMcpServers": ["off"],
                                                                                                 "projects": {str(root): {"mcpServers": {"local": {"url": "http://127.0.0.1:1/mcp"}}, "disabledMcpjsonServers": ["nope"]}}})),
                      ("settings.json", json.dumps({"hooks": {"PreToolUse": [
                          {"matcher": "Write|Edit", "hooks": [{"type": "command", "command": "printf '{\"hookSpecificOutput\": {\"permissionDecision\": \"deny\", \"permissionDecisionReason\": \"trailer\"}}'"}]},
                          {"matcher": "^Bash$", "hooks": [{"type": "command", "command": "printf '{\"hookSpecificOutput\": {\"updatedInput\": {\"command\": \"capped %s\", \"file_path\": \"%s\"}}}' \"$(jq -r .tool_input.command)\" \"$CLAUDE_PROJECT_DIR\""}]},
                          {"matcher": "Read", "hooks": [{"type": "command", "command": "jq -c '{additionalContext: (.tool_name + \" \" + .tool_input.file_path)}'"}]}]}}))):
        (home / rel).parent.mkdir(parents=True, exist_ok=True)
        (home / rel).write_text(text)
    (tmp_path / "cl" / ".claude").mkdir(parents=True)
    (tmp_path / "cl" / ".claude" / "CLAUDE.md").write_text("CLAUDE RULES")
    (root / ".mcp.json").write_text(json.dumps({"mcpServers": {"proj": {"command": "proj-mcp"}, "nope": {"command": "x"}}}))
    (root / ".claude" / "commands").mkdir(parents=True)
    (root / ".claude" / "commands" / "plan.md").write_text("Local plan")
    (root / ".claude" / "settings.local.json").write_text(json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "printf '{\"decision\": \"block\", \"reason\": \"keep going\"}'"}]}]}}))
    on = Settings.from_env({"XDG_CONFIG_HOME": str(tmp_path / "cfg"), "SUPERCLAW_CLAUDE_CONFIG": "1", "CLAUDE_CONFIG_DIR": str(home)})
    assert on.claude_dir == home and on.claude_state == home / ".claude.json" and [p.id for p in on.plugins(root)] == ["hyper"]
    assert on.claude_settings(root) == [home / "settings.json", root / ".claude" / "settings.json", root / ".claude" / "settings.local.json"] and on.claude_settings(root, trusted=False) == [home / "settings.json"]
    assert {s.name: s.description for s in load_skills(on.skill_roots(root))}.get("tidy") == "Tidy. Up." and any(a.name == "reviewer" for a in load_agents(on.agent_roots(root)))
    assert find_command("plan", on.command_roots(root)).template == "Local plan" and find_command("plan", settings.command_roots(root)) is None
    assert "CLAUDE RULES" in project_guidelines(tmp_path / "cl", None, claude=True) and project_guidelines(tmp_path / "cl", None) == ""
    from superclaw.app import build_hooks, mcp_paths
    named = [s.name for s in load_config(mcp_paths(on, root, True)).servers]
    assert named == ["docs", "local", "proj", "hyper"] and [s.name for s in load_config(mcp_paths(on, root, False)).servers] == ["docs", "local", "hyper"] and load_config(mcp_paths(settings, root, True)).servers == []
    claude_dispatch = build_hooks(on, root, True)
    assert build_hooks(settings, root, True) is None and all(h.claude for h in claude_dispatch.hooks) and [h.event for h in claude_dispatch.hooks] == ["beforeTool", "beforeTool", "beforeTool", "stop", "sessionStart", "notification"]
    denied = claude_dispatch.dispatch("beforeTool", {"tool": "edit_file", "args": {"path": "x"}}, "edit_file")
    assert denied.blocked and denied.context == ["trailer"] and not claude_dispatch.dispatch("beforeTool", {"tool": "write_file", "args": {}}, "read_file").blocked
    rewritten = claude_dispatch.dispatch("beforeTool", {"tool": "bash", "args": {"command": "cargo test"}}, "bash")
    assert rewritten.updated_args == {"command": "capped cargo test", "path": str(root)} and not rewritten.blocked
    assert claude_dispatch.dispatch("beforeTool", {"tool": "read_file", "args": {"path": "a.txt"}}, "read_file").context == ["Read a.txt"]
    assert claude_dispatch.dispatch("stop", {"text": "done"}).blocked and claude_dispatch.dispatch("beforeTool", {"tool": "grep", "args": {}}, "grep").updated_args is None
    market = tmp_path / "market"
    (market / ".claude-plugin").mkdir(parents=True)
    shutil.copytree(claude, market / "hyper")
    repo = tmp_path / "gitsrc"
    shutil.copytree(claude, repo / "plugins" / "hyper")
    for command in (["init", "-q", "-b", "main"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "one"], ["tag", "v1"]):
        subprocess.run(["git", "-C", str(repo), *command], check=True, capture_output=True)
    (market / ".claude-plugin" / "marketplace.json").write_text(json.dumps({"name": "Acme", "metadata": {"description": "Acme plugins"}, "plugins": [
        {"name": "hyper", "source": "./hyper"}, {"name": "sub", "source": {"source": "git-subdir", "url": str(repo), "path": "plugins/hyper", "ref": "v1"}},
        {"name": "gh", "source": {"source": "github", "repo": "orkait/hyperstack"}}, {"name": "odd", "source": {"source": "ftp"}}]}))
    assert is_marketplace_ref("hyper@acme") and not is_marketplace_ref("git@github.com:o/r.git") and not is_marketplace_ref(str(market))
    added = add_marketplace(str(market), settings.user_marketplaces, link=True)
    assert added.name == "acme" and added.description == "Acme plugins" and sorted(added.plugins) == ["gh", "hyper", "odd", "sub"] and added.path.is_symlink()
    assert [m.name for m in load_marketplaces(settings.user_marketplaces)] == ["acme"] and load_marketplaces(tmp_path / "nowhere") == []
    assert resolve_source(added, "gh") == ("https://github.com/orkait/hyperstack.git", "", "") and resolve_source(added, "sub") == (str(repo), "v1", "plugins/hyper")
    for spec in ("odd", "missing"):
        with pytest.raises(PluginError):
            resolve_source(added, spec)
    with pytest.raises(PluginError):
        install_from_marketplace("hyper@nope", settings.user_marketplaces, settings.user_plugins)
    fetched = install_from_marketplace("hyper@acme", settings.user_marketplaces, settings.user_plugins)
    assert fetched.id == "hyper" and fetched.path == settings.user_plugins / "hyper" and not fetched.path.is_symlink() and [p.id for p in settings.plugins(root)] == ["hyper"]
    with pytest.raises(PluginError):
        install_from_marketplace("sub@acme", settings.user_marketplaces, settings.user_plugins)
    remove_plugin("hyper", settings.user_plugins)
    assert install_from_marketplace("sub@acme", settings.user_marketplaces, settings.user_plugins).id == "hyper" and (settings.user_plugins / "hyper" / "skills" / "rulebook").is_dir()
    with pytest.raises(PluginError):
        add_marketplace(str(market), settings.user_marketplaces)
    assert remove_marketplace("acme", settings.user_marketplaces) == added.path and market.is_dir() and load_marketplaces(settings.user_marketplaces) == []
    with pytest.raises(PluginError):
        remove_marketplace("acme", settings.user_marketplaces)
    tree = tmp_path / "repo"
    for rel in ("README.md", "pyproject.toml", "src/app/main.py", "src/app/auth/tokens.py", "tests/test_auth.py", "node_modules/x/index.js", "docs/a/b/c/d/e/f/deep.md"):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text("x")
    found = scan_repo(tree, max_depth=4)
    assert "node_modules/x/index.js" not in found.files and "docs/a/b/c/d/e/f/deep.md" not in found.files and found.directories == 4
    assert found.important == ["README.md", "pyproject.toml"] and found.languages[0] == ("python", 3) and not found.truncated
    text = render_repo(found)
    assert text.startswith("Repo: repo\nCounts: files=5 dirs=4\nImportant files: README.md, pyproject.toml\nLanguages: python=3, markdown=1, toml=1") and "  src/app/auth/tokens.py" in text
    assert render_repo(found, budget=80).endswith("...[clipped]") and len(render_repo(found, budget=80).encode()) <= 80 and render_repo(found, budget=0) == ""
    assert [p for p, _ in search_repo(found, "auth tokens")][0] == "src/app/auth/tokens.py" and search_repo(found, "") == []
    assert scan_repo(tree, max_files=2).truncated and len(scan_repo(tree, max_files=2).files) == 2
    mapped = build_system_prompt(PromptInputs(cwd=root, mode=Mode.ASK, model="m", provider="p", repo_map=text))
    assert "<repo_map>" in mapped and "table of contents" in mapped and "src/app/main.py" in mapped
