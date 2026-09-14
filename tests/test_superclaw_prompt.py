import json
from pathlib import Path

import pytest

from superclaw.agents import load_agents
from superclaw.hooks import Dispatcher, load_hooks
from superclaw.mcp import load_config
from superclaw.plugins import PluginError, claude_installed, load_plugins
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
    with pytest.raises(KeyError):
        resolve_agent("nope", [profiles])
    with_agent = build_system_prompt(PromptInputs(cwd=root, mode=Mode.ASK, model="m", provider="p", agent=loaded[0].prompt))
    assert "<agent>" in with_agent and "Only review." in with_agent and "never widens" in with_agent
    assert "<agent>" not in build_system_prompt(PromptInputs(cwd=root, mode=Mode.ASK, model="m", provider="p"))
    equipped = build_system_prompt(PromptInputs(cwd=root, mode=Mode.ASK, model="m", provider="p", tools=("rg", "fd", "jq")))
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
    assert any(s.name == "rulebook" and s.content == "LAWS" for s in load_skills(settings.skill_roots(root))) and any(a.name == "checks" for a in load_agents(settings.agent_roots(root)))
    hooks = load_hooks([(linked.hooks, linked.path)])
    assert [h.event for h in hooks] == ["sessionStart"] and hooks[0].command[:2] == ["/bin/sh", "-c"] and str(linked.path) in hooks[0].command[2] and hooks[0].timeout_s == 5
    dispatcher = Dispatcher(hooks, root)
    assert dispatcher.dispatch("sessionStart", {"session": "s1", "prompt": "hi"}, "startup").context == [f"bootstrap from {linked.path}"]
    assert dispatcher.dispatch("sessionStart", {"session": "s1", "prompt": "hi"}, "resume").context == []
    servers = load_config([linked.mcp]).servers
    assert servers[0].name == "hyper" and servers[0].args == [f"{linked.path}/bin/server.mjs"]
    assert remove_plugin("hyper", settings.user_plugins) == linked.path and claude.is_dir() and settings.plugins(root) == []
    registry = tmp_path / "home" / ".claude" / "plugins"
    registry.mkdir(parents=True)
    (registry / "installed_plugins.json").write_text(json.dumps({"version": 2, "plugins": {"hyper@hyper": [{"scope": "user", "installPath": str(claude)}], "gone@x": [{"installPath": str(tmp_path / "missing")}]}}))
    assert claude_installed(tmp_path / "home") == [claude] and claude_installed(tmp_path / "nohome") == []
    assert settings.claude_plugins is False and Settings.from_env({"SUPERCLAW_CLAUDE_PLUGINS": "1"}).claude_plugins
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
