<div align="center">

# superclaw

**A terminal coding agent whose memory is a graph.**

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](../pyproject.toml)
[![Textual](https://img.shields.io/badge/TUI-Textual%208-4B8BBE?logo=python&logoColor=white)](https://textual.textualize.io)
[![litellm](https://img.shields.io/badge/providers-litellm-111?logo=openai&logoColor=white)](../src/supergraph/ingest/llm/resolve.py)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](../LICENSE)

</div>

superclaw is the harness half of supergraph: a terminal coding agent (TUI and headless `exec`) for people who want an agent that reads, edits, runs and remembers inside one workspace. Every session, plan and durable fact it keeps lives in the supergraph substrate next to it, not in a directory of JSON files, so a run can be resumed, forked and recalled by meaning.

## 🚀 Quickstart

```bash
pip install 'supergraphdb[superclaw]'
export OPENROUTER_API_KEY=...        # or any provider key listed below
cd your-project
superclaw                            # TUI, mode=ask
superclaw --mode auto exec "test_calc.py fails; find the bug, fix it, run pytest -q"
```

First launch downloads the default embedder (model2vec, ~30 MB) into the store.

## ✨ What it does

| Capability | How |
|---|---|
| Edits code | `read_file` `write_file` `edit_file` `list_directory` `glob` `grep`, all jailed to the workspace |
| Runs commands | `bash`, with destructive and network commands classified and gated |
| Plans | `update_plan`, persisted per session and restored on resume |
| Remembers | `memory_search` `memory_note` over the graph, plus automatic recall into every run |
| Loads skills lazily | `SKILL.md` files listed by name and description only; the body loads on `skill` |
| Asks | `ask_user` with options and a recommended default |
| Stays honest | same-error streaks halt the run, empty turns are capped, a headless run is marked `incomplete` when plan items remain |
| Fits the window | proactive compaction at 70% of the context window, summarising the oldest middle and keeping the plan verbatim |

The system prompt is 541 tokens (838 with the confirmation policy).

## 🔐 Permission modes

| Mode | Reads | Writes in workspace | Shell | Destructive or network shell |
|---|---|---|---|---|
| `ask` (default) | allow | prompt | prompt | prompt |
| `auto` | allow | allow | allow | prompt |
| `plan` | allow | hidden | hidden | hidden |
| `unsafe` | allow | allow | allow | allow |

Anything outside the workspace is denied in every mode. Interactive programs (`vim`, `less`, `top`, a bare REPL, `git rebase -i`) are denied because they hang the agent. Headless `exec` has no approver, so a prompt is a denial there; use `--mode auto` or a session grant.

## 🧠 supergraph as the substrate

| superclaw state | Where it lives |
|---|---|
| Sessions and events | namespace `superclaw`: `session:<id>` nodes, `ev:<id>:<seq>` nodes with the payload as the document, `has_event` edges |
| Fork | a new session with the events copied and a `forked_from` edge |
| Plan | `plan` events; the last one is restored on resume |
| Compaction | a `compaction` event; replay substitutes the summary for the events it covered |
| Long-term memory | default namespace: `mem:<sha1>` nodes, recalled with `REMEMBER` before each run |

Nothing superclaw writes into its namespace is visible to plain supergraph queries, and memory notes never leak into the session namespace.

## ⚙️ Configuration

| Setting | Env / flag | Default |
|---|---|---|
| Store path | `SUPERCLAW_DB_PATH`, `--db` | `~/.local/share/superclaw/brain` |
| Model | `SUPERCLAW_MODEL`, `--model` | `openrouter/deepseek/deepseek-v4-flash` |
| Mode | `SUPERCLAW_MODE`, `--mode` | `ask` |
| Context window | `SUPERCLAW_CONTEXT_WINDOW`, `--context-window` | `128000` |
| Turn limit | `--max-turns` | `12` |
| Skills dir | `SUPERCLAW_SKILLS_DIR` | `~/.config/superclaw/skills`, `~/.agents/skills`, `<workspace>/.superclaw/skills` |
| Personal guidelines | `~/.config/superclaw/SUPERCLAW.md` | none |
| Project guidelines | `AGENTS.md`, `SUPERCLAW.md` or `.superclaw/AGENTS.md`, walked from the git root to the cwd | none |

Provider keys follow the model prefix: `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID`, `GOOGLE_AISTUDIO_API_KEY`, `NVIDIA_NIM_API_KEY`, `OLLAMA_API_KEY`. A bare model id routes through OpenRouter.

Guideline files are capped at 8 KiB each and 32 KiB in total; the most specific file wins on conflict.

## 💻 Usage

<details>
<summary>TUI commands</summary>

| Command | Effect |
|---|---|
| `/mode ask\|auto\|plan\|unsafe` | switch the permission mode for the session |
| `/new` | start a fresh session |
| `/sessions` | list recent sessions |
| `/quit` | exit |

Permission prompts answer to `a` (once), `s` (for the session) or `d` (deny).

</details>

<details>
<summary>Headless exec</summary>

```bash
superclaw exec "summarise the failing tests"                      # text
superclaw exec --output-format json "..."                         # one JSON object
superclaw exec --output-format stream-json "..."                  # JSONL events
superclaw --resume latest exec "continue where you left off"
superclaw exec --require-completion "..."                         # exit 2 while plan items remain
echo "prompt on stdin" | superclaw exec -
```

Stream events: `run_start` `usage` `text` `tool_call` `tool_result` `permission_request` `permission_decision` `compaction` `final` `run_end`, each tagged with `schemaVersion` and `runId`.

</details>

<details>
<summary>Skills</summary>

```
~/.config/superclaw/skills/
  run-benchmarks/
    SKILL.md      # ---\nname: run-benchmarks\ndescription: when to use it\n---\ninstructions
```

`superclaw skills` lists what was discovered. Earlier roots win on a name collision; a `SKILL.md` that symlinks outside its root is ignored.

</details>

## 🧭 Design constraints

| Constraint | Consequence |
|---|---|
| Narrow prompts, not one preamble | the core prompt stays under 1k tokens; skills and guidelines load on demand or per project |
| Completion must be able to fail | `--require-completion` refuses a no-tool answer while plan items are pending, nudges three times, then exits 2 |
| No sub-agents by default | there is no delegation tool; fan-out is an extension point, not a feature |

## ⚠️ Limitations

| Limitation | Detail |
|---|---|
| No OS sandbox | the path jail resolves symlinks and refuses escapes, but there is a check-to-use window; treat `unsafe` and `auto` as trusting the workspace |
| No streaming | completions are collected whole, so text appears per turn rather than per token |
| No MCP, no hooks, no LSP | extension points only |
| Substrate gaps | no spend ceiling in `IngestConfig`, no `__origin__` on facts, no `__invalid_at__` window on beliefs |

## ✅ Verification

```
$ .venv/bin/ruff check .
All checks passed!
$ .venv/bin/python -m pytest -q -p no:randomly tests/test_superclaw_*.py
204 passed
$ superclaw --mode auto exec --output-format stream-json "test_calc.py fails. Find the bug in calc.py, fix it, and run pytest -q to prove it passes."
... "type": "tool_call", "name": "edit_file", "args": {"path": "calc.py", "old_string": "return a - b", "new_string": "return a + b"}
... "type": "run_end", "status": "success", "turns": 7, "exitCode": 0
```

## 📄 License

AGPL-3.0, same as supergraph.
