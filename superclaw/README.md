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
| Searches the web | `web_search` (deferred, loads through `tool_search`): Google's Custom Search JSON API when `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` are set, otherwise the open engines through `ddgs` with no key at all (DuckDuckGo first, then Bing, Brave, Yahoo and others, over `primp`'s browser TLS fingerprint, which is what survives bot checks that a plain request trips), with a bare DuckDuckGo request as the last resort when `ddgs` is not installed; `limit` up to 10 and an optional `domains` allowlist that drops other hosts before the model sees them; results come back as numbered title, URL and snippet, redacted and budgeted like every tool output. It is a network tool, so `ask` and `auto` prompt once per query (network always asks below `unsafe`), plan mode hides it, and a headless `exec` has nobody to answer, so it needs `--mode unsafe` for the web tools to run; the same holds for `web_fetch` and `download` |
| Reads pages | `web_fetch` (deferred): a public http or https URL fetched through `primp` impersonating Chrome when installed (the TLS fingerprint a browser has clears most 403s; `old.reddit.com` answers, `www.reddit.com` does not without its API) and `urllib` otherwise, HTML converted to compact markdown (headings, links, lists, code blocks), everything else kept as text, capped at `max_bytes` (256 KiB by default, 2 MiB at most) and 30 s, five redirects each checked again. A page that still answers with an HTTP error is retried through a reader proxy when `SUPERCLAW_READER` is set (`https://r.jina.ai/` renders JavaScript pages to markdown), off by default because it hands the URL to a third party. The page never lands in the main context by default: it is stored whole as a `§ref` that expires after 7 days (working memory, not knowledge), and the model gets a digest, title, size and outline. `prompt` sends a child agent to read the whole page in a fresh context and only its answer comes back; `inline=true` returns the text itself under the normal output budget. Loopback, private, link-local and other special-use hosts are refused even when a public hostname resolves to one, so a fetched page cannot be turned into a probe of this machine; local servers are for `bash` with `curl`, where the network prompt applies |
| Downloads | `download` (deferred): saves a URL into a workspace directory. Video and audio pages go through `yt-dlp` when it is installed (`kind=audio` extracts mp3, playlists are not expanded, names are ASCII-restricted); a URL yt-dlp does not know, or `kind=file`, streams to disk as a plain file named from `Content-Disposition` or the URL. Same public-host guard as `web_fetch`, 256 MiB cap, 10 minute timeout, no overwrites, paths and sizes reported as `changed_files` |
| Uses the host's modern tools | at each run the prompt's `<environment>` names the modern tools present (`rg`, `fd`, `sd`, `xh`, `jq`, `uv`, `bat`, `eza`, `delta`, `hyperfine`, `gh`, `yt-dlp`, `ffmpeg`) and the swaps to prefer in `bash` (`rg` over `grep`, `fd` over `find`, `sd` over `sed`, `xh` over `curl`, `uv` over `pip`); `doctor` lists the same set. The `grep` tool itself runs on `rg` when it is installed (gitignore-aware, hidden files skipped, `--sort path`, the same size cap as `read_file`) and falls back to the Python walker when rg is missing or rejects a pattern it cannot compile, such as a lookbehind |
| Recalls by meaning and by neighbourhood | recall of memories and facts is the substrate's `REMEMBER`, which fuses BM25, the vector signal (`model2vec` 256d by default, so a paraphrase with no shared words still lands) and graph score; `doctor` shows the active embedder and the node and edge counts. Fact recall then expands through the graph: `RECALL FROM <fact> DEPTH 2` pulls the other facts learned from the same page, so one hit brings its siblings, bounded by the recall limit. `/ask <question>` and `superclaw ask` run the substrate's `ANSWER` over facts and stored results with a reader built on the active model: retrieval, a grounded prompt and the cited node ids, no agent loop and no tools |
| Links what it touches | every successful `read_file`, `edit_file` and `write_file` adds a `read` or `wrote` edge from the session to a `file` node; observations link to the session that produced them and facts to the session that learned them. `recall` with a `path`, `/sessions touching <path>` and `superclaw sessions --touching <path>` answer "who worked on this file, what did they learn, what did they store", straight from the edges (`EDGES TO`/`EDGES FROM`), which is what the graph is for |
| Learns facts | knowledge is kept as facts, not pages: `web_fetch` with a `prompt` asks the child to end with a `Facts:` list (`- fact \| exact quote`), and each line becomes a `fact` node asserted through the substrate's belief layer (`ASSERT .. CONFIDENCE .. SOURCE .. EVENT_AT`) with the source URL, the time observed, a confidence and an edge to the page it came from; `memory_note` with `origin: web` and a `source` files one by hand. Facts come back into every prompt in a `<facts>` block with age and URL, into `memory_search` results, and to `/facts [query]`, `/facts as-of DATE`, `/facts retract ID`, or `superclaw facts`. A re-asserted fact refreshes its date, a superseding fact retracts the old one (`RETRACT .. REASON`), and retracted facts vanish from every read, so an as-of query shows what was believed then |
| Remembers | `memory_search` `memory_note` over the graph's memory nodes only (session events, stored prompts and tool results never surface as memories), plus automatic recall into every run. `memory_note` files only what the user stated or selected (`origin`), refuses inferences, secrets and any instruction that would stop a future session raising a concern, and can carry an `expires_days` TTL; recall shows each fact's age and the prompt says a remembered path or flag must be checked before it is recommended |
| Loads skills lazily | `SKILL.md` files listed by name and description only; the body loads on `skill` |
| Calls MCP servers | stdio servers from `mcp.json` connect concurrently under one deadline at launch; each remote tool registers as `mcp_<server>_<tool>` behind the same permission gate as everything else, and deferred, so a 40-tool server costs one line rather than 40 schemas. A server that fails to start, times out or collides on a name is skipped and reported, never fatal |
| Runs as a named profile | an agent profile (`<name>.md` with frontmatter) carries a prompt, a tool allowlist and a model. `--agent` and `/agent` apply it to the session, `delegate(agent=…)` applies it to a child. A profile can only narrow: its tools intersect with `--allow-tools` and it has no way to set the permission mode |
| Takes attachments | `exec -f PATH` and `/attach` inline a text file under the same budget `read_file` uses, or send an image as an image. Attachments resolve through the same jail, so reaching outside the workspace needs `--add-dir` |
| Answers in a shape | `exec --output-schema FILE` hands the model a JSON Schema and checks the final answer against it; a mismatch names the failing path and exits 2, so a caller can pipe `exec` into `jq` |
| Reviews a change | `superclaw review` runs a read-only review of the uncommitted changes, a branch or a commit, forced into plan mode, and prints findings with `file:line`, a severity and a verdict |
| Ships team commands | `<name>.md` under `.superclaw/commands` becomes `/<name>` in the TUI and in `exec`, with `$ARGUMENTS` and `$1`..`$9` expansion, so a repository can check in its own workflows |
| Streams | text arrives as it is generated and the TUI shows the reply growing; tool calls are merged from their deltas, a `<think>` block never leaks token by token, and `stream-json` carries `text_delta` events |
| Runs on a schedule | `superclaw cron add <id> "0 3 * * *" --prompt "..."` stores a job as a graph node; `cron run` is a foreground scheduler that fires due jobs as their own sessions titled `cron <id>`, `--once` fires what is due and exits (for an external scheduler), and jobs that became due while nothing was running skip to their next slot unless `--catch-up` |
| Drafts before it builds | `exec --spec "task"` runs read-only, inspects the workspace, and calls `submit_spec` with a concrete plan (goal, files, steps, tests, risks, out of scope) saved under `.superclaw/specs`; the run stops there with exit 3. `superclaw spec list\|show\|approve <id> [--note]` implements an approved spec in the current mode as its own session |
| Verifies | `superclaw verify` detects the workspace's checks (`go test`, `package.json` scripts through the lockfile's package manager, pytest, `cargo test`), runs them with a per-check timeout and prints pass/fail with the failing tail; `--attempts N` hands each failure to the agent as a fix-it turn and reruns, never weakening a check to pass |
| Knows the layout | a repo map (counts, important files, languages, paths) rides the system prompt from turn one, budgeted, so the model reads the right file instead of listing directories first; `superclaw repo-map --query auth tokens` ranks paths the same way |
| Serves editors | `superclaw acp` speaks the Agent Client Protocol over stdio, so Zed and other ACP clients drive the same loop: sessions, streamed replies, tool calls with status and diffs, and permission prompts routed to the editor as `session/request_permission` |
| Asks | `ask_user` with options and a recommended default |
| Stays honest | same-error streaks halt the run, empty turns are capped, identical calls warn at 3 and 42 calls in one turn warn, a final message that promises more work is sent back once, and `--verify` runs a read-only verifier call that must return `{passed, reason, nextAction}` before a headless run counts as done |
| Fits the window | pressure is measured against the model's real window minus a 16,384-token reserve, anchored on the provider's reported usage rather than a local estimate. Under pressure the harness first prunes older tool results (over 8,192 chars) to a head and tail with no model call, and only if that is not enough summarises everything before the last 20,000 tokens, never cutting between a tool call and its result. The summariser gets a projection that keeps every user message verbatim, assistant text, the last eight tool calls per turn, errors and edits, plus the previous summary; it must answer in nine fixed sections; the plan, loaded skills and edited files ride along verbatim and the model is told to continue without acknowledging the summary. Prunes and compactions are session events, so a resumed session replays the same shortened context |

The system prompt is 846 tokens (1,160 with the confirmation policy), plus the profile body when `--agent` is used. Only seven tool schemas ride every request (`read_file` `edit_file` `write_file` `grep` `bash` `python` `tool_search`); the other twelve are listed one line each and load on demand through `tool_search`, so a first turn is about 2.3k tokens before the user's message (eager=1210 all=2630 prompt=1125 first_turn=2335, measured with the ink-quarter estimator).

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
| Store path | `SUPERCLAW_DB_PATH`, `--db` | `~/.local/share/superclaw/brain`; shared between concurrent superclaw processes: the first one owns it and serves the rest over a unix socket in the runtime dir; `/doctor` in a session shows `(owner)` or `(attached)` |
| Model | `SUPERCLAW_MODEL`, `--model`, `/model` in the TUI | `openrouter/deepseek/deepseek-v4-flash`; ids are `provider/slug` for `openrouter`, `groq`, `cerebras`, `ollama` (cloud), `aistudio`, `nvidia_nim`, `opencode` (OpenCode Zen). `/model` and `superclaw models` list what each connected provider serves (prices shown as `$input/output` per million tokens): the provider's live `/models` endpoint (public for OpenRouter and NVIDIA, keyed elsewhere) cached for a day under `~/.cache/superclaw/models`, merged with the bundled catalog for context windows and prices, with embedding, audio, image and moderation models filtered out. A model only the live list knows still gets its window and price from that list |
| Fallback models | `SUPERCLAW_FALLBACK_MODELS`, `--fallback-model a,b` | none; tried in order when the main model errors, and every turn starts again at the main model |
| Streaming | `SUPERCLAW_STREAM=0` turns it off | on; text is forwarded as it arrives, tool-call deltas are merged by index, and anything inside a `<think>` block is withheld until the block closes. The final `text` event, the transcript and `exec` text and json output are unchanged; `stream-json` gains `text_delta` events |
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
| Extra roots | `--add-dir PATH` (repeatable) | none; a granted directory is readable, writable and bound into the sandbox for `bash` and the kernel, and stops counting as an out-of-workspace write |
| Worktree | `-w/--worktree [NAME]`, `--worktree-dir` | off; creates or reuses `<data dir>/worktrees/superclaw-worktree-<repo>-<hash>/<name>` on branch `superclaw/<name>`, default name `task-<utc timestamp>` |
| MCP servers | `~/.config/superclaw/mcp.json`, plus `<workspace>/.superclaw/mcp.json` with `--trust-workspace` | none; `{"mcpServers": {"docs": {"command": "…", "args": [], "env": {}}, "hosted": {"url": "https://…", "headers": {}}}}`, the shape claude and cursor already use. stdio and streamable HTTP; the legacy SSE transport is refused. `superclaw mcp` and `/mcp` list what connected and what was skipped |
| Agent profiles | `--agent NAME`, `/agent` in the TUI | none; `<workspace>/.superclaw/agents/<name>.md` then `~/.config/superclaw/agents/<name>.md`, frontmatter `name` `description` `tools` `model` over a prompt body. `superclaw agents` lists them |
| Repo map | `SUPERCLAW_REPO_MAP=0` turns it off; `superclaw repo-map [--json\|--query TEXT]` prints it | on; a deterministic map of the workspace (`git ls-files` when in a repository, else a walk that skips build and cache directories) rendered as counts, the files that usually matter, languages and paths, clamped to 6,000 bytes and placed in the system prompt as `<repo_map>` so the model knows the layout on turn one. `--query` ranks paths by term |
| Web search | `SUPERCLAW_SEARCH=google\|duckduckgo`, `GOOGLE_API_KEY`, `GOOGLE_CSE_ID` | `google` when both Google values are set, `duckduckgo` otherwise; DuckDuckGo needs nothing but rate-limits bursts, so requests are spaced 1.5 s apart and a refusal is retried once after 3 s before the error names the Google variables |
| Reader proxy | `SUPERCLAW_READER=https://r.jina.ai/` | off; when set, a page that answers with an HTTP error is fetched again as `<reader><url>` and the output says `Via:` |
| Plugins | `superclaw plugin list\|install <dir\|git url>\|remove <id>` | none; a plugin is a directory with a `plugin.json` (`id`, `description`, `version`) that bundles any of `skills/`, `agents/`, `commands/`, `hooks.json` and `mcp.json`. User plugins live under `~/.config/superclaw/plugins`, project plugins under `<workspace>/.superclaw/plugins`; skills, agents and commands from a project plugin load always, its hooks and MCP servers only with `--trust-workspace`, the same rule as the bare workspace files |
| User commands | `/<name> args` in the TUI, `superclaw exec "/<name> args"` | none; `<workspace>/.superclaw/commands/<name>.md` then `~/.config/superclaw/commands/<name>.md`, frontmatter `description` `agent` `model` over a template. `superclaw commands` lists them |
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
| Transcript | `❯` user lines, assistant markdown, one card per tool call: status glyph (`◐` running, `✓`, `✗`), tool name, target (path, pattern, command, task), then the body - a unified diff for `edit_file`/`write_file`, output lines for everything else, `§id` when stored. The head carries the outcome (`+a -b` for edits, `N lines` for everything else) and the call's seconds. Bodies are folded by default; diffs and failures show their first 12 lines. Click a card to unfold it, or press `ctrl+o` to unfold every card, current and future, and again to fold them. Child (`delegate`) calls are indented |
| Working line | one row while a run is in flight, hidden otherwise: spinner, the real phase, then elapsed seconds (excluding time spent in a prompt), tool count and tokens so far. Phases: `recalling` while the graph is searched, `thinking` with the recall overview (`2 memories · 14 skills · repo map 40 files · 12 earlier messages`, counts only, never content), `writing` once text streams, the tool name with its target while a tool runs, `waiting for you` during a permission prompt, `cancelling` after `esc` |
| Title bar and status bar | workspace, branch and session id above; `● mode`, context reading `◔ 21.4K/1.0M · 2.1%`, run cost and tokens kept out below; the model sits on the composer border. Segments drop as the terminal narrows (tiers at 58, 80 and 100 columns) |
| Command palette | typing `/` lists matching commands with usage and help; `up`/`down` move the highlight, `tab` or enter on a highlighted row picks it, enter with nothing highlighted runs what was typed, `esc` closes it. A command that takes arguments is completed into the prompt; one without runs at once |

| Command | Effect |
|---|---|
| `/mode ask\|auto\|plan\|unsafe` | switch the permission mode for the session |
| `/model [list\|id]` | no argument opens the picker: recent models first, then one group per connected provider, type to filter, enter picks. `list` prints the same rows. An id switches at once, fuzzy when unique (`/model v4-pro`), and a model on a provider without a key opens setup for that provider. The choice is saved as `SUPERCLAW_MODEL` |
| `/effort low\|medium\|high\|off` | set the model's reasoning effort for the session, saved as `SUPERCLAW_EFFORT` and shown in the status bar |
| `/<name> [args]` | a user command from `.superclaw/commands`: the template expands (`$ARGUMENTS`, `$1`..`$9` shell-split, `$$`) and runs as a prompt; history keeps the typed line. Builtins win on a name collision |
| `/agent [name\|none]` | no argument lists the profiles and marks the active one; a name applies its prompt and narrows the tools; `none` clears it and restores the allowlist the command line asked for |
| `/attach <path>` | attach a file or image and leave its `[File #n]` or `[Image #n]` marker in the composer; the markers are the source of truth, as in Claude Code: when you send, the attachments whose markers are still in the text go with the message in marker order, and a marker you deleted takes its attachment out. A marker is one unit, like a chip in Claude Code: backspace, delete, ctrl+w, ctrl+u or ctrl+k touching any part of it removes the whole marker, and typing inside one lands after it, so a broken half-marker can never sit in the composer. The typed line, markers included, is what lands in history |
| drop, paste, `ctrl+v` | dropping a file onto the terminal pastes its path (plain, quoted or `file://`); superclaw attaches it and leaves `[Image #1]` or `[File #1]` in the composer. A paste of 3+ lines or over 2,000 chars becomes `[Pasted text #1 +N lines]` with the text attached whole instead of being cut to its first line. `ctrl+v` reads the clipboard the way Claude Code does: an image (`xclip` or `wl-paste` on Linux, `pngpaste` or `osascript` on macOS) is saved under `~/.local/share/superclaw/clipboard` and attached as `[Image #n]`, so a screenshot taken to the clipboard is one keypress away; text goes through the same paste rules |
| `/mcp` | the MCP tools that connected, the servers that were skipped and why, and any config problem |
| `/new` (aliases `/clear`, `/reset`), `/resume [id\|latest]`, `/fork [id\|latest]`, `/sessions [query]` | session lifecycle; `/new` starts a fresh session with an empty context and a zeroed status bar, as `/clear` does in Claude Code, and the old session stays resumable; `/fork` copies a session (the current one by default) into a new one and continues it; `/sessions <query>` searches stored events by meaning instead of listing |
| `/usage` | tokens and cost spent in this session |
| `/context [prompt]` | what the next request costs, by category |
| `/facts [query\|as-of DATE [query]\|retract ID [reason]]` | facts learned from sources with their age and URL; `as-of` shows what was believed on a date |
| `/ask <question>` | answer from stored facts and results through the substrate's reader; prints the cited ids |
| `/recall <§id\|query>` | bring back or search stored tool results, rendered as a card |
| `/compact` | summarize older turns into one message now, freeing the window before the next run; a session event, so it survives resume |
| `/retry` | run the last prompt again |
| `/rename <title>` | name the session; the title shows in the title bar and `/sessions` |
| `/export` | write the transcript to `superclaw-transcript-<id>.md` in the workspace |
| `/tools` | list every tool, its side effect, and whether it is hidden in the current mode |
| `/permissions` | show the mode, the session tool grants and the remembered command prefixes |
| `/doctor` | terminal, sandbox, mode, agent, model, store, MCP, the graph's embedder with node and edge counts, the modern host tools present, and connected-provider health, read-only |
| `/setup` | connect a provider key and model without leaving the TUI |
| `/help`, `/exit` | housekeeping; `/quit` is an alias of `/exit`, and a bare `exit`, `quit`, `:q`, `:q!`, `:wq` or `:wq!` on its own line leaves too, as in Claude Code |

Keys: `up` and `down` recall earlier prompts into the composer (a shell-style history seeded from the session, so it survives resume; a saved draft returns when you step back past the newest); while the command palette is open the same keys move its highlight. `shift+tab` cycles the permission mode through `ask`, `auto` and `plan`; `ctrl+o` unfolds every tool card and folds them again; `esc` cancels the current run at the next tool boundary (the result records `cancelled`); `ctrl+c` cancels a running turn first and quits on the second press, also from inside a permission, question or setup dialog, where cancelling closes the dialog as a deny. Permission prompts answer to `a` (once), `s` (for the session), `p` (remember the offered prefix, only shown when the model offered one) or `d` (deny). `/new`, `/resume` and `/clear` wait for the run to finish. A provider failure ends the turn with a `run failed` line plus one next step (a rejected key points at `/setup`, an unknown model at `/model`, a full window at `/new`, rate limits and unreachable hosts say so) and leaves the shell open; `superclaw exec` prints the same line with the command-line equivalents.

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
superclaw exec -f docs/spec.md -f shot.png "does the UI match?"   # attach a file, an image, or both
superclaw exec --output-schema review.json "review the diff"      # final answer must match the schema, or exit 2
superclaw --agent reviewer exec "review the last commit"          # run under a profile
superclaw -w exec "risky refactor"                                # in a throwaway worktree on superclaw/task-<ts>
superclaw --add-dir ../shared exec "port the helper across"       # a second writable root
superclaw exec "/pr 42"                                            # a user command from .superclaw/commands
superclaw review --base main "focus on error handling"            # read-only review: findings with file:line and a verdict
superclaw review --commit HEAD                                     # or --uncommitted (default)
superclaw export -o session.json && superclaw import session.json # move a session between stores or machines
echo "prompt on stdin" | superclaw exec -
```

Stream events: `run_start` `context` (memories, skills, repo files, earlier messages and prompt tokens the run starts with) `usage` `text_delta` `text` `tool_call` `tool_result` `permission_request` `permission_decision` `compaction` `budget` `final` `run_end`, each tagged with `schemaVersion` and `runId`. `text_delta` carries each fragment as it arrives; `text` still carries the whole turn. `usage` carries `input_tokens` `output_tokens` `cache_read_tokens` `cost_usd` `run_cost_usd` `context_used` `context_window` `saved_tokens` `kept_out_tokens`; `run_end` carries `savedTokens` and `keptOutTokens`. Child events carry `child: <session id>`.

`superclaw context [prompt]` prints what the first request would cost by category (system prompt, guidelines, skills index, memory recall, tool schemas, history) against the resolved window. `superclaw models [--provider name] [--refresh]` prints every model each connected provider serves, one row per model with the context window, tool support, price per million tokens and whether the row came from the live list or the bundled catalog; `--refresh` ignores the day-old cache. `superclaw usage` prints per-session call, token and cost totals. `superclaw sessions [query]` lists recent sessions, or searches their stored events by meaning and exits 1 when nothing matches. `--fork <id|latest>` copies a session into a new one and continues from it, alongside `--resume`.

`superclaw update` says how superclaw is installed and whether something newer exists: an editable checkout is compared with its upstream branch, a `uv tool` or pip install with PyPI. `--apply` runs the matching command (`git pull --ff-only`, `uv tool upgrade supergraphdb`, or `pip install --upgrade supergraphdb`). `superclaw export [id|latest] [-o FILE]` writes a session and its events as versioned JSON; `superclaw import FILE|-` creates a new session that replays them, with `parent` pointing at the source id and `cwd` rewritten to this workspace, so the copy can be resumed, forked and searched like any other. `superclaw doctor`, `mcp`, `agents`, `skills`, `commands`, `plugin`, `repo-map`, `update`, `spec list|show` and `verify` (with one attempt) never open the store, so they still answer while a session holds it. The commands that do need it (`exec`, `sessions`, `usage`, `context`, the TUI) report the lock and point at `--db <path>` rather than raising.

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

<details>
<summary>Agent profiles</summary>

```
<workspace>/.superclaw/agents/reviewer.md
~/.config/superclaw/agents/reviewer.md
```

```markdown
---
name: reviewer
description: Reviews a diff and reports findings, never edits.
tools: read_file, grep, glob, bash
model: openrouter/some-model
---
You review code. Report findings with file:line. Never edit files.
```

`superclaw agents` lists them, `--agent reviewer` runs the session under one, `/agent` switches mid-session, and `delegate(task, agent="reviewer")` hands the role to a child. The workspace directory wins on a name collision, which is safe in one direction only: a profile narrows the tools and cannot touch the permission mode, so a repository can ship one without handing itself an escalation.

</details>

<details>
<summary>User commands</summary>

```
<workspace>/.superclaw/commands/pr.md      # project, checked in and shared
~/.config/superclaw/commands/pr.md         # personal
```

```markdown
---
description: Open a PR for the current branch.
agent: reviewer
---
Open a pull request titled "$1". Summary: $ARGUMENTS
```

`/pr "fix build" touches three files` expands the template and runs it as a prompt: `$ARGUMENTS` is the raw argument string, `$1`..`$9` are shell-split positionals (so a quoted argument stays whole), `$$` is a literal dollar, and a template with no placeholder gets the arguments appended. `agent:` and `model:` route the run the way `/agent` and `/model` do. Names are `[a-z0-9-]` only and builtins win, so a stray file cannot shadow `/mode`. `superclaw commands` lists them; `superclaw exec "/pr 42"` works headless.

</details>

<details>
<summary>Review</summary>

`superclaw review` picks a diff (`--uncommitted` by default, `--base BRANCH`, or `--commit SHA`), runs the reviewer prompt in plan mode so the model can read any file for context but never edit, and prints findings as `file:line`, `blocker` / `should-fix` / `nit`, problem, fix, then a `Verdict:` line. The diff goes through the same diff-aware budget the tool boundary uses, the prompt says when it was cut, and untracked files are named rather than inlined. An optional argument focuses the reviewer; an empty change exits with `nothing to review`.

</details>

<details>
<summary>Editor backend (ACP)</summary>

`superclaw acp` is a JSON-RPC 2.0 peer over newline-delimited stdio, the wire shape the Agent Client Protocol specifies, so an editor can run superclaw as its agent. Everything else goes to stderr.

| Method | superclaw behaviour |
|---|---|
| `initialize` | protocol version 1, `loadSession`, image and embedded-context prompts, `session/list` |
| `session/new` `{cwd}` | a new session in the store; `cwd` must be the absolute directory this process serves (start it there or with `-C`), and the editor's `mcpServers` are ignored because superclaw owns its own MCP config |
| `session/load` | replays the transcript as `user_message_chunk` and `agent_message_chunk` updates |
| `session/list`, `session/set_mode` | recent sessions; `ask` `auto` `plan` `unsafe` |
| `session/prompt` | text, image and resource blocks become the prompt; replies stream as `agent_message_chunk`, tool calls arrive as `tool_call` then `tool_call_update` with `read` `edit` `search` `execute` `think` kinds and the diff or output as content; returns `end_turn`, `cancelled` or `max_turn_requests`. One prompt at a time per process |
| `session/cancel` | a notification; the run stops at the next tool boundary |
| `session/request_permission` | sent to the client with `allow_once`, `allow_always` (session, and the command prefix when one was offered) and `reject_once` options that map one-to-one onto the TUI's `a` `s` `p` `d`; a cancelled outcome denies and cancels the run |

</details>

<details>
<summary>MCP servers</summary>

```json
{
  "mcpServers": {
    "docs":   { "command": "npx", "args": ["-y", "some-mcp-server"], "env": {"API_KEY": "..."} },
    "hosted": { "url": "https://mcp.example.com/mcp", "headers": {"Authorization": "Bearer ..."} },
    "off":    { "command": "other", "disabled": true }
  }
}
```

`~/.config/superclaw/mcp.json` always, `<workspace>/.superclaw/mcp.json` only with `--trust-workspace`, because an entry there is a command this process runs. Servers connect concurrently under one deadline, so startup costs the slowest server rather than the sum, and a server that fails to start, times out or collides on a tool name is skipped with a reason instead of taking the launch down. Tools arrive as `mcp_<server>_<tool>`, deferred behind `tool_search`, gated as network side effects. A `command` entry is stdio; a `url` entry is streamable HTTP: JSON-RPC is POSTed with `Accept: application/json, text/event-stream`, the `Mcp-Session-Id` the server hands back on `initialize` is sent on every later request, and a response delivered as an SSE body is unwrapped to the message with the matching id. The legacy SSE transport (`"type": "sse"`) is refused with a reason, as is an entry that mixes the two shapes.

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
| MCP is tools only | stdio and streamable HTTP transports, no legacy SSE stream, no OAuth (put a token in `headers`), and resources and prompts are not consumed |
| One writer per store | supergraph takes an exclusive lock per path, so the first superclaw to open the brain owns it and serves it over a unix socket under `$XDG_RUNTIME_DIR/superclaw/<hash of the store path>.sock`; every later superclaw on the same store attaches and sends its queries there. If the owner exits, the next query from an attached session takes the lock and starts serving. A superclaw older than this scheme holding the lock without a socket is reported, with the fix being to restart it |
| Images do not survive resume | an attached image rides the live run; the session event records only how many there were, because base64 in the event document would land in the same FTS index `REMEMBER` and `superclaw sessions <query>` search |
| CLI-only surfaces | `review`, `verify`, `spec`, `cron`, `export`/`import`, `plugin` and `repo-map` have no slash command yet; the TUI reaches them through `exec "/name"` user commands or a second terminal |
| ACP is prompts and tools | `fs/read_text_file`, `fs/write_text_file` and `terminal/*` are not served; superclaw reads and writes locally through its own jail |
| No LSP | extension point only |
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
