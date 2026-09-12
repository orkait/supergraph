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
| Edits code | `read_file` `write_file` `edit_file` `list_directory` `glob` `grep`, all jailed to the workspace; an edit or overwrite fails unless the file was read this session and is unchanged on disk since. `read_file` pages at 2,000 lines with a continuation offset and clips lines past 2,000 chars; `grep` skips binaries and files over 256 KiB |
| Never re-reads what it holds | the harness tracks every file range the model has seen, keyed by content hash and by the message that carried it. A `read_file` for a range still in the context returns a one-line pointer instead of the bytes (`force=true` overrides); an edit, an external change, a pruned or compacted message, or a truncated result drops the claim, so the next read is real |
| Keeps tool output inside the window | every result crosses one boundary: redact, then a token budget per output category (file, search, test, process, diff) that keeps the lines that matter - both ends of a file range, one hit per file, failure lines with context, distinct errors plus the tail - then the full redacted text is saved under `~/.local/share/superclaw/artifacts/<session>/` and the model view cites the path, the original size and what survived. Hook feedback passes the same boundary. Edits carry a unified diff for the TUI that never enters the model context |
| Labels what it did not write | tool output arrives in `<untrusted source=…>` blocks the prompt ranks below the user; secrets (API keys, tokens, JWTs, private keys, auth headers, `*_password=` values) are scrubbed at the tool boundary before the model sees them |
| Runs commands | `bash` inside a `bubblewrap` sandbox: read-only root, writable workspace and `/tmp`, no network, `~/.ssh` `~/.aws` `~/.gnupg` masked; destructive and network commands classified and gated; `require_escalated` with a `justification` runs on the host after approval |
| Plans | `update_plan`, persisted per session and restored on resume |
| Remembers | `memory_search` `memory_note` over the graph, plus automatic recall into every run. `memory_note` files only what the user stated or selected (`origin`), refuses inferences, secrets and any instruction that would stop a future session raising a concern, and can carry an `expires_days` TTL; recall shows each fact's age and the prompt says a remembered path or flag must be checked before it is recommended |
| Loads skills lazily | `SKILL.md` files listed by name and description only; the body loads on `skill` |
| Asks | `ask_user` with options and a recommended default |
| Stays honest | same-error streaks halt the run, empty turns are capped, identical calls warn at 3 and 42 calls in one turn warn, a final message that promises more work is sent back once, and `--verify` runs a read-only verifier call that must return `{passed, reason, nextAction}` before a headless run counts as done |
| Fits the window | pressure is measured against the model's real window minus a 16,384-token reserve, anchored on the provider's reported usage rather than a local estimate. Under pressure the harness first prunes older tool results (over 8,192 chars) to a head and tail with no model call, and only if that is not enough summarises everything before the last 20,000 tokens, never cutting between a tool call and its result. The summariser gets a projection that keeps every user message verbatim, assistant text, the last eight tool calls per turn, errors and edits, plus the previous summary; it must answer in nine fixed sections; the plan, loaded skills and edited files ride along verbatim and the model is told to continue without acknowledging the summary. Prunes and compactions are session events, so a resumed session replays the same shortened context |

The system prompt is 541 tokens (838 with the confirmation policy). Only six tool schemas ride every request (`read_file` `edit_file` `write_file` `grep` `bash` `tool_search`); the rest are listed one line each and load on demand through `tool_search`, so a first turn is about 1.6k tokens before the user's message (eager=964 all=2035 prompt=873 first_turn=1837).

## 🔐 Permission modes

| Mode | Reads | Writes in workspace | Shell | Destructive or network shell |
|---|---|---|---|---|
| `ask` (default) | allow | prompt | prompt | prompt |
| `auto` | allow | allow | allow | prompt |
| `plan` | allow | hidden | hidden | hidden |
| `unsafe` | allow | allow | allow | allow |

Anything outside the workspace is denied in every mode. Interactive programs (`vim`, `less`, `top`, a bare REPL, `git rebase -i`) are denied because they hang the agent. Headless `exec` has no approver, so a prompt is a denial there; use `--mode auto` or a session grant. Without `bwrap` on the host, `auto` shell degrades to a prompt rather than running unsandboxed. An approval can be remembered as a command prefix (`p` in the TUI) when the model offered a narrow `prefix_rule`; prefixes for `rm`, `sudo`, interpreters, single tokens and heredoc commands are never remembered.

## 🧠 supergraph as the substrate

| superclaw state | Where it lives |
|---|---|
| Sessions and events | namespace `superclaw`: `session:<id>` nodes, `ev:<id>:<seq>` nodes with the payload as the document, `has_event` edges |
| Fork | a new session with the events copied and a `forked_from` edge |
| Plan | `plan` events; the last one is restored on resume |
| Compaction | a `compaction` event; replay substitutes the summary for the events it covered |
| Prompt and failures | every run logs a `prompt` event (hash, token count, full text) so what the model saw is reconstructable; a resumed session reports `prompt_drift` when the rebuilt prompt differs; provider failures are `error` events |
| Long-term memory | default namespace: `mem:<sha1>` nodes, recalled with `REMEMBER` before each run |

Nothing superclaw writes into its namespace is visible to plain supergraph queries, and memory notes never leak into the session namespace.

## ⚙️ Configuration

| Setting | Env / flag | Default |
|---|---|---|
| Store path | `SUPERCLAW_DB_PATH`, `--db` | `~/.local/share/superclaw/brain` |
| Model | `SUPERCLAW_MODEL`, `--model` | `openrouter/deepseek/deepseek-v4-flash` |
| Mode | `SUPERCLAW_MODE`, `--mode` | `ask` |
| Context window | `SUPERCLAW_CONTEXT_WINDOW`, `--context-window` | `0` = resolved from the bundled model catalog (1,000,000 for the default model); `128000` when the model is unknown |
| Turn limit | `--max-turns` | `12` |
| Token budget | `SUPERCLAW_BUDGET_TOKENS`, `--budget-tokens` | `0` (unlimited); a run stops as `incomplete` once spent |
| Spend budget | `SUPERCLAW_BUDGET_USD`, `--budget-usd` | `0` (unlimited); priced per call from the catalog, cached input at the cache-read rate |
| Every tunable | `superclaw/settings.py` `Limits` | one frozen dataclass holds every threshold, clamp, budget and preview width; nothing else in the package carries a literal |
| Hooks | `~/.config/superclaw/hooks.json`, plus `<workspace>/.superclaw/hooks.json` with `--trust-workspace` | off until the file says `"enabled": true`; events `sessionStart` `beforeTool` `afterTool` `stop`, regex `matcher` on the tool name, JSON payload on stdin, exit 2 blocks a tool or asks the run to continue, stdout `{"additionalContext": ...}` is injected |
| Intent gate | `--intent-gate` | off; one narrow model call classifies the request as `answer`, `diagnose`, `change` or `monitor`, and `answer` hides writes, shell and network while `diagnose` hides writes |
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
superclaw exec --verify "..."                                     # plus a verifier call that refuses proxy signals
echo "prompt on stdin" | superclaw exec -
```

Stream events: `run_start` `usage` `text` `tool_call` `tool_result` `permission_request` `permission_decision` `compaction` `budget` `final` `run_end`, each tagged with `schemaVersion` and `runId`. `usage` carries `input_tokens` `output_tokens` `cache_read_tokens` `cost_usd` `run_cost_usd` `context_used` `context_window`.

`superclaw context [prompt]` prints what the first request would cost by category (system prompt, guidelines, skills index, memory recall, tool schemas, history) against the resolved window.

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
| Completion must be able to fail | `--require-completion` refuses a no-tool answer while plan items are pending; `--verify` adds a separate read-only model call that treats passing tests, a finished plan and visible effort as evidence only when they cover every requirement; three nudges, then exit 2 |
| No sub-agents by default | there is no delegation tool; fan-out is an extension point, not a feature |

## ⚠️ Limitations

| Limitation | Detail |
|---|---|
| Linux-only sandbox | `bubblewrap` covers `bash`; file tools rely on the path jail, which resolves symlinks but has a check-to-use window. No macOS Seatbelt yet, and network approval is all-or-nothing rather than a domain allowlist |
| No streaming | completions are collected whole, so text appears per turn rather than per token |
| No MCP, no LSP | extension points only |
| Substrate gaps | no spend ceiling in `IngestConfig`, no `__origin__` on facts, no `__invalid_at__` window on beliefs |

## ✅ Verification

```
$ .venv/bin/ruff check .
All checks passed!
$ .venv/bin/python -m pytest -q -p no:randomly tests/test_superclaw_*.py
37 passed
$ superclaw --mode auto exec --output-format stream-json "test_calc.py fails. Find the bug in calc.py, fix it, and run pytest -q to prove it passes."
... "type": "tool_call", "name": "edit_file", "args": {"path": "calc.py", "old_string": "return a - b", "new_string": "return a + b"}
... "type": "run_end", "status": "success", "turns": 7, "exitCode": 0
```

## 📄 License

AGPL-3.0, same as supergraph.
