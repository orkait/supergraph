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
superclaw setup                      # stores a provider key in ~/.config/superclaw/credentials.env (or set OPENROUTER_API_KEY)
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
| Keeps tool output inside the window | every result crosses one boundary: redact, then a token budget per output category (file, search, test, process, diff) that keeps the lines that matter - both ends of a file range, one hit per file, failure lines with context, distinct errors plus the tail - then the full redacted text becomes an `obs:<id>` node in the substrate and the model view carries `§id`, the original size and what survived. Hook feedback passes the same boundary. Edits carry a unified diff for the TUI that never enters the model context |
| Loses nothing to pressure | pruning replaces the middle of an old result with `recall §id to expand`; `recall` returns any stored result in 4,000-token chunks, or searches every stored result by meaning (`REMEMBER` over the `obs` nodes) when the id is unknown. Recalled chunks are ordinary results, so they are pruned again when pressure returns |
| Computes instead of reading | `python` is a persistent kernel per session, sandboxed like `bash`, whose variables survive between calls; only what the model prints crosses the boundary. Inside it `obs("§id")` loads any stored result as text, `sh(cmd)` runs a command and returns `Run(out, code, lines)`, `query(dsl)` runs a read-only supergraph query. `bash` with `capture="r"` keeps the whole output in the kernel and in the substrate and shows the model a five-line tail |
| Keeps children out of its context | `delegate(task, refs, files)` runs a child loop with a fresh context that shares the workspace, tools and stored results but never the parent's conversation; the child's session is its own node with a `parent` edge, its tokens and cost are added to the parent's totals, and the parent receives a short result, the changed files and the `§refs` of what the child stored, so it can `recall` a finding without re-reading. Depth is capped at 2 and a child budget is floored at 20,000 tokens so it can never die on its own prompt |
| Reports what it kept out | every `usage` event and `run_end` carry `saved_tokens` (budgeted, pruned and captured output that never reached the window) and `kept_out_tokens` (everything children spent in their own windows); the TUI shows the sum in the status bar and the done line |
| Labels what it did not write | tool output arrives in `<untrusted source=…>` blocks the prompt ranks below the user; secrets (API keys, tokens, JWTs, private keys, auth headers, `*_password=` values) are scrubbed at the tool boundary before the model sees them |
| Runs commands | `bash` inside a `bubblewrap` sandbox: read-only root, writable workspace and `/tmp`, no network, `~/.ssh` `~/.aws` `~/.gnupg` masked; destructive and network commands classified and gated; `require_escalated` with a `justification` runs on the host after approval |
| Plans | `update_plan`, persisted per session and restored on resume |
| Remembers | `memory_search` `memory_note` over the graph, plus automatic recall into every run. `memory_note` files only what the user stated or selected (`origin`), refuses inferences, secrets and any instruction that would stop a future session raising a concern, and can carry an `expires_days` TTL; recall shows each fact's age and the prompt says a remembered path or flag must be checked before it is recommended |
| Loads skills lazily | `SKILL.md` files listed by name and description only; the body loads on `skill` |
| Asks | `ask_user` with options and a recommended default |
| Stays honest | same-error streaks halt the run, empty turns are capped, identical calls warn at 3 and 42 calls in one turn warn, a final message that promises more work is sent back once, and `--verify` runs a read-only verifier call that must return `{passed, reason, nextAction}` before a headless run counts as done |
| Fits the window | pressure is measured against the model's real window minus a 16,384-token reserve, anchored on the provider's reported usage rather than a local estimate. Under pressure the harness first prunes older tool results (over 8,192 chars) to a head and tail with no model call, and only if that is not enough summarises everything before the last 20,000 tokens, never cutting between a tool call and its result. The summariser gets a projection that keeps every user message verbatim, assistant text, the last eight tool calls per turn, errors and edits, plus the previous summary; it must answer in nine fixed sections; the plan, loaded skills and edited files ride along verbatim and the model is told to continue without acknowledging the summary. Prunes and compactions are session events, so a resumed session replays the same shortened context |

The system prompt is 809 tokens (1,106 with the confirmation policy). Only seven tool schemas ride every request (`read_file` `edit_file` `write_file` `grep` `bash` `python` `tool_search`); the other nine are listed one line each and load on demand through `tool_search`, so a first turn is about 2.3k tokens before the user's message (eager=1210 all=2630 prompt=1125 first_turn=2335, measured with the ink-quarter estimator).

## 🔐 Permission modes

| Mode | Reads | Writes in workspace | Shell | Destructive or network shell |
|---|---|---|---|---|
| `ask` (default) | allow | prompt | prompt | prompt |
| `auto` | allow | allow | allow | prompt |
| `plan` | allow | hidden | hidden | hidden |
| `unsafe` | allow | allow | allow | allow |

`shift+tab` cycles `ask`, `auto` and `plan` in the TUI (the status bar shows the current mode); `unsafe` is deliberately not in that cycle. Reach it with `/mode unsafe`, `--mode unsafe`, or `--dangerously-skip-permissions`, which is the same as `--mode unsafe` and only belongs in a sandbox you can discard. Anything outside the workspace is denied in every mode. Interactive programs (`vim`, `less`, `top`, a bare REPL, `git rebase -i`) are denied because they hang the agent. Headless `exec` has no approver, so a prompt is a denial there; use `--mode auto` or a session grant. Without `bwrap` on the host, `auto` shell degrades to a prompt rather than running unsandboxed. An approval can be remembered as a command prefix (`p` in the TUI) when the model offered a narrow `prefix_rule`; prefixes for `rm`, `sudo`, interpreters, single tokens and heredoc commands are never remembered.

## 🧠 supergraph as the substrate

| superclaw state | Where it lives |
|---|---|
| Sessions and events | namespace `superclaw`: `session:<id>` nodes, `ev:<id>:<seq>` nodes with the payload as the document, `has_event` edges |
| Fork | a new session with the events copied and a `forked_from` edge |
| Tool results | `obs:<id>` nodes with the full redacted output as the document, id = hash of tool, call and body; the window carries `§id` |
| Children | `delegate` runs get their own `session:<id>` with `parent` set to the caller; their events, prunes and results are addressable like any other session |
| Plan | `plan` events; the last one is restored on resume |
| Compaction | a `compaction` event; replay substitutes the summary for the events it covered |
| Prompt and failures | every run logs a `prompt` event (hash, token count, full text) so what the model saw is reconstructable; a resumed session reports `prompt_drift` when the rebuilt prompt differs; provider failures are `error` events |
| Long-term memory | default namespace: `mem:<sha1>` nodes, recalled with `REMEMBER` before each run |

Nothing superclaw writes into its namespace is visible to plain supergraph queries, and memory notes never leak into the session namespace.

## ⚙️ Configuration

| Setting | Env / flag | Default |
|---|---|---|
| Provider key | `superclaw setup [--provider openrouter\|groq\|cerebras\|ollama\|aistudio\|nvidia_nim\|opencode]`, `/setup` in the TUI, or the provider's env var | saved to `~/.config/superclaw/credentials.env` (mode 600) together with `SUPERCLAW_MODEL`; the environment overrides the file. The TUI opens without a key and shows the setup screen; `exec` refuses to run without one |
| Store path | `SUPERCLAW_DB_PATH`, `--db` | `~/.local/share/superclaw/brain` |
| Model | `SUPERCLAW_MODEL`, `--model`, `/model` in the TUI | `openrouter/deepseek/deepseek-v4-flash`; ids are `provider/slug` for `openrouter`, `groq`, `cerebras`, `ollama` (cloud), `aistudio`, `nvidia_nim`, `opencode` (OpenCode Zen). `/model` and `superclaw models` list what each connected provider serves (prices shown as `$input/output` per million tokens): the provider's live `/models` endpoint (public for OpenRouter and NVIDIA, keyed elsewhere) cached for a day under `~/.cache/superclaw/models`, merged with the bundled catalog for context windows and prices, with embedding, audio, image and moderation models filtered out. A model only the live list knows still gets its window and price from that list |
| Mode | `SUPERCLAW_MODE`, `--mode` | `ask` |
| Reasoning effort | `SUPERCLAW_EFFORT`, `/effort low\|medium\|high\|off` in the TUI | off; when set it is sent as `reasoning_effort` on every call and shown in the status bar. litellm drops the parameter for models that do not support it, so it is a no-op there rather than an error |
| Context window | `SUPERCLAW_CONTEXT_WINDOW`, `--context-window` | `0` = resolved from the bundled model catalog (1,000,000 for the default model); `128000` when the model is unknown |
| Turn limit | `--max-turns` | `12` |
| Token budget | `SUPERCLAW_BUDGET_TOKENS`, `--budget-tokens` | `0` (unlimited); a run stops as `incomplete` once spent |
| Spend budget | `SUPERCLAW_BUDGET_USD`, `--budget-usd` | `0` (unlimited); priced per call from the catalog, cached input at the cache-read rate |
| Glyphs | `SUPERCLAW_ASCII=1`, or a locale without `UTF-8` in `LC_ALL`, `LC_CTYPE` or `LANG` | Unicode set `❯ ◐ ✓ ✗ · ◔ ● ↳ … →`, rounded borders and the block wordmark; every glyph is in DejaVu Sans Mono, the `Monospace` alias on Ubuntu. The ASCII set `> ~ + x | # * -> ... ->` with plain borders takes over when the locale cannot carry them. superclaw never installs fonts or changes terminal settings |
| Every tunable | `superclaw/settings.py` `Limits` and `Glyphs` | one frozen dataclass holds every threshold, clamp, budget and preview width, another every drawn symbol; nothing else in the package carries a literal |
| Hooks | `~/.config/superclaw/hooks.json`, plus `<workspace>/.superclaw/hooks.json` with `--trust-workspace` | off until the file says `"enabled": true`; events `sessionStart` `beforeTool` `afterTool` `stop`, regex `matcher` on the tool name, JSON payload on stdin, exit 2 blocks a tool or asks the run to continue, stdout `{"additionalContext": ...}` is injected |
| Tool exposure | `--allow-tools`, `--deny-tools` (comma or space separated) | expose only the named tools, or hide the named tools, on top of the mode's own visibility |
| Intent gate | `--intent-gate` | off; one narrow model call classifies the request as `answer`, `diagnose`, `change` or `monitor`, and `answer` hides writes, shell and network while `diagnose` hides writes |
| Skills dir | `SUPERCLAW_SKILLS_DIR` | `~/.config/superclaw/skills`, `~/.agents/skills`, `<workspace>/.superclaw/skills` |
| Personal guidelines | `~/.config/superclaw/SUPERCLAW.md` | none |
| Project guidelines | `AGENTS.md`, `SUPERCLAW.md` or `.superclaw/AGENTS.md`, walked from the git root to the cwd | none |

Provider keys follow the model prefix: `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID`, `GOOGLE_AISTUDIO_API_KEY`, `NVIDIA_NIM_API_KEY`, `OLLAMA_API_KEY`, `OPENCODE_API_KEY` (OpenCode Zen, base `https://opencode.ai/zen/v1`). When it is unset, the key is read from OpenCode's own `auth.json` under `~/.local/share/opencode` (honoring `OPENCODE_AUTH_PATH`, `OPENCODE_DIR` and `XDG_DATA_HOME`); that is a Go subscription key, so it routes to `https://opencode.ai/zen/go/v1` with the required `x-opencode-session` header instead of the pay-per-use Zen endpoint, and a machine already logged in through the OpenCode CLI needs no extra setup. `OPENCODE_API_BASE` overrides the endpoint and `OPENCODE_SESSION` the routing id. A bare model id routes through OpenRouter.

Guideline files are capped at 8 KiB each and 32 KiB in total; the most specific file wins on conflict.

## 💻 Usage

<details>
<summary>TUI commands</summary>

`superclaw` with no subcommand opens the TUI (it refuses a non-TTY stdin and points at `exec`). Without a key it opens anyway and shows the provider setup screen: pick a provider, paste the key, then pick a model from the list that provider serves. It draws with the glyph set from the configuration table, which any stock monospace font carries; `SUPERCLAW_ASCII=1` switches to plain ASCII. The welcome screen shows the version, workspace, branch and model; the first prompt replaces it with the transcript.

| Surface | What it shows |
|---|---|
| Transcript | `❯` user lines, assistant markdown, one card per tool call: status glyph (`◐` running, `✓`, `✗`), tool name, target (path, pattern, command, task), then the body - a unified diff for `edit_file`/`write_file`, output lines for everything else, `§id` when stored. Bodies fold at 12 lines; click to expand. Child (`delegate`) calls are indented |
| Working line | spinner, current phase (`thinking`, or the tool name), elapsed seconds excluding time spent in a permission prompt, tool count |
| Title bar and status bar | workspace, branch and session id above; `● mode`, context reading `◔ 21.4K/1.0M · 2.1%`, run cost and tokens kept out below; the model sits on the composer border. Segments drop as the terminal narrows (tiers at 58, 80 and 100 columns) |
| Command palette | typing `/` lists matching commands with usage and help; `up`/`down` move the highlight, `tab` or enter on a highlighted row picks it, enter with nothing highlighted runs what was typed, `esc` closes it. A command that takes arguments is completed into the prompt; one without runs at once |

| Command | Effect |
|---|---|
| `/mode ask\|auto\|plan\|unsafe` | switch the permission mode for the session |
| `/model [list\|id]` | no argument opens the picker: recent models first, then one group per connected provider, type to filter, enter picks. `list` prints the same rows. An id switches at once, fuzzy when unique (`/model v4-pro`), and a model on a provider without a key opens setup for that provider. The choice is saved as `SUPERCLAW_MODEL` |
| `/effort low\|medium\|high\|off` | set the model's reasoning effort for the session, saved as `SUPERCLAW_EFFORT` and shown in the status bar |
| `/new`, `/resume [id\|latest]`, `/fork [id\|latest]`, `/sessions` | session lifecycle; `/fork` copies a session (the current one by default) into a new one and continues it |
| `/usage` | tokens and cost spent in this session |
| `/context [prompt]` | what the next request costs, by category |
| `/recall <§id\|query>` | bring back or search stored tool results, rendered as a card |
| `/compact` | summarize older turns into one message now, freeing the window before the next run; a session event, so it survives resume |
| `/retry` | run the last prompt again |
| `/rename <title>` | name the session; the title shows in the title bar and `/sessions` |
| `/export` | write the transcript to `superclaw-transcript-<id>.md` in the workspace |
| `/tools` | list every tool, its side effect, and whether it is hidden in the current mode |
| `/permissions` | show the mode, the session tool grants and the remembered command prefixes |
| `/doctor` | terminal, sandbox, model and connected-provider health, read-only |
| `/setup` | connect a provider key and model without leaving the TUI |
| `/clear`, `/help`, `/quit` | housekeeping |

Keys: `up` and `down` recall earlier prompts into the composer (a shell-style history seeded from the session, so it survives resume; a saved draft returns when you step back past the newest); while the command palette is open the same keys move its highlight. `shift+tab` cycles the permission mode through `ask`, `auto` and `plan`; `esc` cancels the current run at the next tool boundary (the result records `cancelled`); `ctrl+c` cancels a running turn first and quits on the second press, also from inside a permission, question or setup dialog, where cancelling closes the dialog as a deny. Permission prompts answer to `a` (once), `s` (for the session), `p` (remember the offered prefix, only shown when the model offered one) or `d` (deny). `/new`, `/resume` and `/clear` wait for the run to finish. A provider failure ends the turn with a `run failed` line plus one next step (a rejected key points at `/setup`, an unknown model at `/model`, a full window at `/new`, rate limits and unreachable hosts say so) and leaves the shell open; `superclaw exec` prints the same line with the command-line equivalents.

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

Stream events: `run_start` `usage` `text` `tool_call` `tool_result` `permission_request` `permission_decision` `compaction` `budget` `final` `run_end`, each tagged with `schemaVersion` and `runId`. `usage` carries `input_tokens` `output_tokens` `cache_read_tokens` `cost_usd` `run_cost_usd` `context_used` `context_window` `saved_tokens` `kept_out_tokens`; `run_end` carries `savedTokens` and `keptOutTokens`. Child events carry `child: <session id>`.

`superclaw context [prompt]` prints what the first request would cost by category (system prompt, guidelines, skills index, memory recall, tool schemas, history) against the resolved window. `superclaw models [--provider name] [--refresh]` prints every model each connected provider serves, one row per model with the context window, tool support, price per million tokens and whether the row came from the live list or the bundled catalog; `--refresh` ignores the day-old cache. `superclaw usage` prints per-session call, token and cost totals. `--fork <id|latest>` copies a session into a new one and continues from it, alongside `--resume`.

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
| Children never share the parent's context | `delegate` is the only fan-out; a child gets a task, handles and a budget, and returns a short result. Nothing a child reads enters the parent's window |

## ⚠️ Limitations

| Limitation | Detail |
|---|---|
| Linux-only sandbox | `bubblewrap` covers `bash` and the `python` kernel; without it both degrade to a prompt in `auto`. File tools rely on the path jail, which resolves symlinks but has a check-to-use window. No macOS Seatbelt yet, and network approval is all-or-nothing rather than a domain allowlist |
| Kernel checkpoints are picklable-only | the kernel namespace is checkpointed into the substrate after every run and restored on `--resume`, but values that cannot be pickled (lambdas, open handles, live modules) are dropped; a kernel timeout resets the live namespace and the next run restores the last checkpoint |
| No streaming | completions are collected whole, so text appears per turn rather than per token |
| No MCP, no LSP | extension points only |
| Substrate gaps | no spend ceiling in `IngestConfig`, no `__origin__` on facts, no `__invalid_at__` window on beliefs |

## ✅ Verification

```
$ .venv/bin/ruff check .
All checks passed!
$ .venv/bin/python -m pytest -q -p no:randomly tests/test_superclaw_*.py
15 passed
$ superclaw --mode auto exec --output-format stream-json "test_calc.py fails. Find the bug in calc.py, fix it, and run pytest -q to prove it passes."
... "type": "tool_call", "name": "edit_file", "args": {"path": "calc.py", "old_string": "return a - b", "new_string": "return a + b"}
... "type": "run_end", "status": "success", "turns": 7, "exitCode": 0
```

## 📄 License

AGPL-3.0, same as supergraph.
