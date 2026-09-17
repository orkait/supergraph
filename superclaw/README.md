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
curl -LsSf https://raw.githubusercontent.com/orkait/supergraph/main/install.sh | sh
superclaw setup                      # stores a provider key in ~/.config/superclaw/credentials.env (or set OPENROUTER_API_KEY)
cd your-project
superclaw                            # TUI, mode=ask
superclaw --mode auto exec "test_calc.py fails; find the bug, fix it, run pytest -q"
```

The installer brings uv if it is missing, then installs superclaw and supergraph as one tool. `--ref <branch|tag>` pins a revision, `--local <checkout>` installs editable from a clone, `--python` picks the interpreter (3.10 to 3.14, default 3.13). `supergraphdb` is not on PyPI, so `pip install` does not work; `superclaw update` compares an existing install with its source and `--apply` upgrades it.

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
| Borrows Claude Code | `claude(task, model, allow_edits, session)` runs the installed Claude Code headless in the same workspace, on the account already signed in on this host - the only sanctioned way to spend a Claude subscription from another tool, since Anthropic stopped honouring subscription OAuth tokens in third-party harnesses in April 2026. It brings its own context, model and tools; `--strict-mcp-config` keeps this machine's MCP servers out of it, it gets `Read` `Grep` `Glob` `WebSearch` `WebFetch` until `allow_edits=true` hands it `acceptEdits`, and anything that would prompt is denied rather than left hanging. The result carries its turns, its cost, a `§ref` to the full answer and a session id to continue that same conversation on a later call. Capped at 900 s and $5 |
| Reports what it kept out | every `usage` event and `run_end` carry `saved_tokens` (budgeted, pruned and captured output that never reached the window) and `kept_out_tokens` (everything children spent in their own windows); the TUI shows the sum in the status bar and the done line |
| Labels what it did not write | tool output arrives in `<untrusted source=…>` blocks the prompt ranks below the user; secrets (API keys, tokens, JWTs, private keys, auth headers, `*_password=` values) are scrubbed at the tool boundary before the model sees them |
| Runs commands | `bash` inside a `bubblewrap` sandbox: read-only root, writable workspace and `/tmp`, no network, `~/.ssh` `~/.aws` `~/.gnupg` masked, and the container sockets (`docker`, `podman`, `containerd`, `crio`) replaced with `/dev/null` so a command cannot mount the host through a container and step around the sandbox's own rules; destructive and network commands classified and gated; `require_escalated` with a `justification` runs on the host after approval |
| Knows its own sandbox | the system prompt's environment block states what bash can and cannot do: no network, a fresh `/tmp` and `/dev/shm` on every call, only the workspace persists, the workspace ignores `chmod 0700`, and the masked credential paths. Discovering these by trial and error cost one real session about twenty turns |
| Never blocks on a slow command | `bash(run_in_background=true)` answers at once with a job id and keeps the command running; `bash_output(id)` returns whatever is new since the last read, `bash_output()` lists every job with its state, `bash_output(id, kill=true)` stops one. Output is drained by a reader thread, so a pipeline that buffers (`\| tail`) no longer hides progress, and a foreground command still ends at `timeout_ms` (default 60 s, max 600 s). Jobs are killed when the runtime closes |
| Plans | `update_plan`, persisted per session and restored on resume |
| Searches the web | `web_search` (deferred, loads through `tool_search`): Google's Custom Search JSON API when `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` are set, otherwise the open engines through `ddgs` with no key at all (DuckDuckGo first, then Bing, Brave, Yahoo and others, over `primp`'s browser TLS fingerprint, which is what survives bot checks that a plain request trips), with a bare DuckDuckGo request as the last resort when `ddgs` is not installed; `limit` up to 10 and an optional `domains` allowlist that drops other hosts before the model sees them; results come back as numbered title, URL and snippet, redacted and budgeted like every tool output. It is a network tool, so `ask` and `auto` prompt once per query (network always asks below `unsafe`), plan mode hides it, and a headless `exec` has nobody to answer, so it needs `--mode unsafe` for the web tools to run; the same holds for `web_fetch` and `download` |
| Reads pages | `web_fetch` (deferred): a public http or https URL fetched through `primp` impersonating Chrome when installed (the TLS fingerprint a browser has clears most 403s; `old.reddit.com` answers, `www.reddit.com` does not without its API) and `urllib` otherwise, HTML converted to compact markdown (headings, links, lists, code blocks), everything else kept as text, capped at `max_bytes` (256 KiB by default, 2 MiB at most) and 30 s, five redirects each checked again. A page that still answers with an HTTP error is retried through a reader proxy when `SUPERCLAW_READER` is set (`https://r.jina.ai/` renders JavaScript pages to markdown), off by default because it hands the URL to a third party. The page never lands in the main context by default: it is stored whole as a `§ref` that expires after 7 days (working memory, not knowledge), and the model gets a digest, title, size and outline. `prompt` sends a child agent to read the whole page in a fresh context and only its answer comes back; `inline=true` returns the text itself under the normal output budget. Loopback, private, link-local and other special-use hosts are refused even when a public hostname resolves to one, so a fetched page cannot be turned into a probe of this machine; local servers are for `bash` with `curl`, where the network prompt applies |
| Downloads | `download` (deferred): saves a URL into a workspace directory. Video and audio pages go through `yt-dlp` when it is installed (`kind=audio` extracts mp3, playlists are not expanded, names are ASCII-restricted); a URL yt-dlp does not know, or `kind=file`, streams to disk as a plain file named from `Content-Disposition` or the URL. Same public-host guard as `web_fetch`, 256 MiB cap, 10 minute timeout, no overwrites, paths and sizes reported as `changed_files` |
| Uses the host's modern tools | at each run the prompt's `<environment>` names the modern tools present (`rg`, `fd`, `sd`, `xh`, `jq`, `uv`, `bat`, `eza`, `delta`, `hyperfine`, `gh`, `yt-dlp`, `ffmpeg`) and the swaps to prefer in `bash` (`rg` over `grep`, `fd` over `find`, `sd` over `sed`, `xh` over `curl`, `uv` over `pip`); `doctor` lists the same set. The `grep` tool itself runs on `rg` when it is installed (gitignore-aware, hidden files skipped, `--sort path`, the same size cap as `read_file`) and falls back to the Python walker when rg is missing or rejects a pattern it cannot compile, such as a lookbehind |
| Maintains itself | `superclaw maintain` and `/maintain` run the substrate's `SYS EXPIRE` (drops web pages and documents past their TTL), decay stale facts (a fact not re-asserted for 90 days halves its confidence; below 0.2 it is retracted with a reason), `SYS OPTIMIZE`, and print `SYS HEALTH`; the same pass minus optimize runs by itself at launch when the last one is older than a day, and `doctor` shows tombstones, string bloat, dead vectors and the last run. Before the first `unsafe` run of a session the store takes a `SYS SNAPSHOT`; `/snapshots` lists them and `/rollback NAME` restores one and opens a fresh session on the restored store. Snapshots live in the memory of the process that owns the store (probed: a fresh process sees none), so rollback is a safety net for the running TUI, not a backup; `superclaw export` is the backup. Not wired, and why: `SYS CONSOLIDATE` and `SYS CONTRADICTIONS` work on entity edges that free-text facts do not carry yet, `SYS CRON` is not configured in the embedded store, and `EVOLVE` rules act on store metrics rather than nodes |
| Keeps documents on demand | `ingest` (deferred) parses a workspace file through the substrate's `INGEST` router into a `document` node plus searchable `chunk` nodes: markdown, text, html, csv, json, docx, pptx, xlsx and images need nothing extra, pdf needs `supergraphdb[ingest]`, audio needs `supergraphdb[audio]`, and the error names the extra when one is missing. Chunks expire after 30 days unless `pin` is true, so a kept document is a decision, not a side effect of reading. `download` with `keep: true` ingests what it saved. `recall` with a query lists matching chunks next to stored results, `recall doc=<id> chunk=<n>` reads a document in order, and `/ask` answers over chunks as well as facts and results |
| Recalls by meaning and by neighbourhood | recall of memories and facts is the substrate's `REMEMBER`, which fuses BM25, the vector signal (`model2vec` 256d by default, so a paraphrase with no shared words still lands) and graph score; `doctor` shows the active embedder and the node and edge counts. Fact recall then expands through the graph: `RECALL FROM <fact> DEPTH 2` pulls the other facts learned from the same page, so one hit brings its siblings, bounded by the recall limit. `/ask <question>` and `superclaw ask` run the substrate's `ANSWER` over facts and stored results with a reader built on the active model: retrieval, a grounded prompt and the cited node ids, no agent loop and no tools |
| Links what it touches | every successful `read_file`, `edit_file` and `write_file` adds a `read` or `wrote` edge from the session to a `file` node; observations link to the session that produced them and facts to the session that learned them. `recall` with a `path`, `/sessions touching <path>` and `superclaw sessions --touching <path>` answer "who worked on this file, what did they learn, what did they store", straight from the edges (`EDGES TO`/`EDGES FROM`), which is what the graph is for |
| Learns facts | knowledge is kept as facts, not pages: `web_fetch` with a `prompt` asks the child to end with a `Facts:` list (`- fact \| exact quote`), and each line becomes a `fact` node asserted through the substrate's belief layer (`ASSERT .. CONFIDENCE .. SOURCE .. EVENT_AT`) with the source URL, the time observed, a confidence and an edge to the page it came from; `memory_note` with `origin: web` and a `source` files one by hand. Facts come back into every prompt in a `<facts>` block with age and URL, into `memory_search` results, and to `/facts [query]`, `/facts as-of DATE`, `/facts retract ID`, or `superclaw facts`. A re-asserted fact refreshes its date, a superseding fact retracts the old one (`RETRACT .. REASON`), and retracted facts vanish from every read, so an as-of query shows what was believed then |
| Remembers | `memory_search` `memory_note` over the graph's memory nodes only (session events, stored prompts and tool results never surface as memories), plus automatic recall into every run. `memory_note` files only what the user stated or selected (`origin`), refuses inferences, secrets and any instruction that would stop a future session raising a concern, and can carry an `expires_days` TTL; recall shows each fact's age and the prompt says a remembered path or flag must be checked before it is recommended |
| Loads skills lazily | `SKILL.md` files listed by name and description only; the body loads on `skill` |
| Ships that brain as a Claude plugin | `integrations/claude` is a Claude Code plugin: the MCP entry above plus a skill that tells Claude when to search, when to file and what is refused. `claude plugin marketplace add orkait/supergraph` then `claude plugin install superclaw-memory@orkait`, or `--plugin-dir <checkout>/integrations/claude` for one session. Its tools arrive as `mcp__plugin_superclaw-memory_superclaw__*`. `--strict-mcp-config` suppresses plugin servers, so do not pass it when you want them |
| Refuses to connect to itself | superclaw reads Claude's MCP config, so installing that plugin points superclaw at `superclaw mcp serve`: a second process attaching to the brain it already owns and re-registering the memory tools it already has. Such an entry is skipped with a reason instead of connected |
| Lends its brain to other agents | `superclaw mcp serve` is an MCP stdio server that hands `memory_search`, `memory_note` and `recall` to any MCP client over the same store a live session is using: it attaches through the shared-store socket when another superclaw owns the brain, and owns it when none does. The same `Tool` objects back both front ends, so the schemas cannot drift. Writes land under a session titled `another agent over MCP`, which `superclaw sessions` lists like any other. `claude mcp add superclaw -- superclaw mcp serve` puts Claude Code on this machine's memory. The brain is machine-wide, not per project, so that client can `recall` a stored result from any workspace superclaw has worked in; `superclaw --db <path> mcp serve` serves a separate store instead |
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
| Always interruptible | the provider call carries superclaw's own deadline rather than trusting the client library's, so a provider that accepts the request and never answers fails at `completion_timeout_s` with the "provider unreachable" hint instead of hanging. `esc` is observed between streamed chunks, between tool calls, at the top of every turn and while a permission dialog waits, and it kills a running `bash` process group instead of waiting out its timeout |
| Nothing blocks the interface | every command that touches the model, the graph or the repository runs on a worker with the working line showing what it is doing: `/ask`, `/context`, `/doctor`, `/maintain` and `/compact`. Commands that would change the rules underneath a live run, including `/mode`, `/effort`, `/rollback`, `/maintain`, `/setup` and `shift+tab`, wait for it to finish |
| Fits the window | pressure is measured against the smaller of 60% of the model's real window and the window minus a 16,384-token reserve, anchored on the provider's reported usage rather than a local estimate. The share matters on a large window: with a 1,000,000-token model the reserve alone would put the trigger at 983,616, so a 140,000-token conversation would never compact and every turn would re-send all of it. Under pressure the harness first prunes older tool results (over 8,192 chars) to a head and tail with no model call, and only if that is not enough summarises everything before the last 20,000 tokens, never cutting between a tool call and its result. The summariser gets a projection that keeps every user message verbatim, assistant text, the last eight tool calls per turn, errors and edits, plus the previous summary; it must answer in nine fixed sections; the plan, loaded skills and edited files ride along verbatim and the model is told to continue without acknowledging the summary. Prunes and compactions are session events, so a resumed session replays the same shortened context |
| Flushes state before it forgets | when the window is under pressure and a memory tool is present, one silent turn runs first with only `memory_note` and `update_plan` exposed, so durable findings reach the graph before the detail becomes a summary. It answers `NOTHING` when there is nothing worth keeping, emits a `flush` event either way, and is capped at 6 model calls. OpenClaw calls this a pre-compaction state flush; Claude Code forces a checkpoint at 50,000 tokens |

The system prompt is 846 tokens (1,160 with the confirmation policy), plus the profile body when `--agent` is used. Only seven tool schemas ride every request (`read_file` `edit_file` `write_file` `grep` `bash` `python` `tool_search`); the other thirteen are listed one line each and load on demand through `tool_search`, so a first turn is about 2.3k tokens before the user's message (eager=1210 all=2630 prompt=1125 first_turn=2335, measured with the ink-quarter estimator).

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
| Compaction | a `compaction` event; replay substitutes the summary for the events it covered. The summary is nine fixed sections (task, every user message verbatim, decisions, files, commands, errors, constraints and security rules verbatim, open items, next step), ends with the session id so `recall` can restore any §ref, and `preCompact` hooks can add instructions |
| Prompt and failures | every run logs a `prompt` event (hash and token count, with the text itself stored once per distinct prompt) so what the model saw is reconstructable; a resumed session reports `prompt_drift` when the rebuilt prompt differs; provider failures are `error` events |
| Session identity | a session records its working directory, git branch and a running byte total, and takes its title from the first prompt sent to it, so the resume picker can show what a session was about rather than an id |
| Long-term memory | default namespace: `mem:<sha1>` nodes, recalled with `REMEMBER` before each run |

Nothing superclaw writes into its namespace is visible to plain supergraph queries, and memory notes never leak into the session namespace.

## ⚙️ Configuration

| Setting | Env / flag | Default |
|---|---|---|
| Provider key | `superclaw setup [--provider openrouter\|groq\|cerebras\|ollama\|aistudio\|nvidia_nim\|opencode]`, `/setup` in the TUI, or the provider's env var | saved to `~/.config/superclaw/credentials.env` (mode 600) together with `SUPERCLAW_MODEL`; the environment overrides the file. The TUI opens without a key and shows the setup screen; `exec` refuses to run without one |
| Store path | `SUPERCLAW_DB_PATH`, `--db` | `~/.local/share/superclaw/brain`; shared between concurrent superclaw processes: the first one owns it and serves the rest over a unix socket in the runtime dir; `/doctor` in a session shows `(owner)` or `(attached)` |
| Stores each system prompt once | a run records only the prompt's hash and token count; the text itself is a `prompt:<hash>` node written once and shared by every run that sends the same prompt, recoverable with `SessionStore.prompt_text`. On the live brain 60 runs had stored 1,052,449 bytes of system prompt across 29 distinct prompts, 37% of all event bytes, to satisfy a drift check that only ever reads the hash |
| Model | `SUPERCLAW_MODEL`, `--model`, `/model` in the TUI | `openrouter/deepseek/deepseek-v4-flash`; ids are `provider/slug` for `openrouter`, `groq`, `cerebras`, `ollama` (cloud), `aistudio`, `nvidia_nim`, `opencode` (OpenCode Zen). `/model` and `superclaw models` list what each connected provider serves (prices shown as `$input/output` per million tokens): the provider's live `/models` endpoint (public for OpenRouter and NVIDIA, keyed elsewhere) cached for a day under `~/.cache/superclaw/models`, merged with the bundled catalog for context windows and prices, with embedding, audio, image and moderation models filtered out. A model only the live list knows still gets its window and price from that list |
| Fallback models | `SUPERCLAW_FALLBACK_MODELS`, `--fallback-model a,b` | none; tried in order when the main model errors, and every turn starts again at the main model |
| Streaming | `SUPERCLAW_STREAM=0` turns it off | on; text is forwarded as it arrives, tool-call deltas are merged by index, and anything inside a `<think>` block is withheld until the block closes. The final `text` event, the transcript and `exec` text and json output are unchanged; `stream-json` gains `text_delta` events |
| Mode | `SUPERCLAW_MODE`, `--mode` | `ask` |
| Reasoning effort | `SUPERCLAW_EFFORT`, `/effort low\|medium\|high\|off` in the TUI | off; when set it is sent as `reasoning_effort` on every call and shown in the status bar. litellm drops the parameter for models that do not support it, so it is a no-op there rather than an error |
| Context window | `SUPERCLAW_CONTEXT_WINDOW`, `--context-window` | `0` = resolved from the bundled model catalog (1,000,000 for the default model); `128000` when the model is unknown |
| Output cap | `SUPERCLAW_MAX_OUTPUT_TOKENS` | `0` = the smaller of the model's catalog `max_output_tokens` and 32,768 (8,192 for an unknown model), and each turn sends the smaller of that and the free window (`window - used`, floor 1,024) so a full context never overflows. Hidden reasoning counts against the cap on DeepSeek-style models, so a 4,096 cap left whole turns with no visible text; a turn that ends `length` with no text now stops the run at once with `stop_reason: max_tokens` (ACP `max_tokens`) and a message naming the cap and the two remedies, `/effort` and the variable. On Anthropic models `/effort` also sets the thinking budget (litellm maps low, medium, high to 1,024, 2,048, 4,096 and caps it under the output cap) |
| Turn limit | `--max-turns N` | off; a run ends when the model answers without tools, or when a guard fires (same-error streak, identical calls, promise nudges, `--budget-tokens`, `--budget-usd`). Set `N` to force a final answer after that many turns, which headless callers may want |
| Token budget | `SUPERCLAW_BUDGET_TOKENS`, `--budget-tokens` | `0` (unlimited); a run stops as `incomplete` once spent |
| Spend budget | `SUPERCLAW_BUDGET_USD`, `--budget-usd` | `0` (unlimited); priced per call from the catalog, cached input at the cache-read rate |
| Glyphs | `SUPERCLAW_ASCII=1`, or a locale without `UTF-8` in `LC_ALL`, `LC_CTYPE` or `LANG` | Unicode set `❯ ◐ ✓ ✗ · ◔ ● ↳ … →`, rounded borders and the block wordmark; every glyph is in DejaVu Sans Mono, the `Monospace` alias on Ubuntu. The ASCII set `> ~ + x | # * -> ... ->` with plain borders takes over when the locale cannot carry them. superclaw never installs fonts or changes terminal settings |
| Every tunable | `superclaw/settings.py` `Limits` and `Glyphs` | one frozen dataclass holds every threshold, clamp, budget and preview width, another every drawn symbol; nothing else in the package carries a literal |
| Hooks | `~/.config/superclaw/hooks.json`, plus `<workspace>/.superclaw/hooks.json` with `--trust-workspace` | off until the file says `"enabled": true`; events `sessionStart` `userPrompt` `beforeTool` `afterTool` `permissionRequest` `notification` `preCompact` `stop` `subagentStop` `sessionEnd`, regex `matcher` on the tool name (or the notification type, compaction trigger, session-end reason), JSON payload on stdin, exit 2 blocks a tool or asks the run (or a delegate child) to continue, stdout `{"additionalContext": ...}` is injected, `{"hookSpecificOutput": {"updatedInput": {...}}}` rewrites the tool's arguments, `{"permission": "allow"|"deny"}` on `permissionRequest` answers instead of the user, and `preCompact` context lands in the summariser's instructions. A file in Claude's nested schema is read as the Claude hook dialect below |
| Tool exposure | `--allow-tools`, `--deny-tools` (comma or space separated) | expose only the named tools, or hide the named tools, on top of the mode's own visibility |
| Intent gate | `--intent-gate` | off; one narrow model call classifies the request as `answer`, `diagnose`, `change` or `monitor`, and `answer` hides writes, shell and network while `diagnose` hides writes |
| Extra roots | `--add-dir PATH` (repeatable) | none; a granted directory is readable, writable and bound into the sandbox for `bash` and the kernel, and stops counting as an out-of-workspace write |
| Worktree | `-w/--worktree [NAME]`, `--worktree-dir` | off; creates or reuses `<data dir>/worktrees/superclaw-worktree-<repo>-<hash>/<name>` on branch `superclaw/<name>`, default name `task-<utc timestamp>` |
| MCP servers | `superclaw mcp add NAME -- CMD ARGS...`, `mcp add NAME --url URL [--header K=V]`, `mcp remove NAME`, each with `--scope user\|project`; the files are `~/.config/superclaw/mcp.json` and `<workspace>/.superclaw/mcp.json`, the latter read only with `--trust-workspace` | none; `{"mcpServers": {"docs": {"command": "…", "args": [], "env": {}}, "hosted": {"url": "https://…", "headers": {}}}}`, the shape claude and cursor already use. stdio and streamable HTTP; the legacy SSE transport is refused. `superclaw mcp` and `/mcp` list what connected and what was skipped |
| Agent profiles | `--agent NAME`, `/agent` in the TUI | two built in, `explore` and `review`, both read-only; then `<workspace>/.superclaw/agents/<name>.md` then `~/.config/superclaw/agents/<name>.md`, frontmatter `name` `description` `tools` `model` over a prompt body. A role directory one level down (`agents/<role>/PROFILE.md`, the layout Claude-format plugins such as hyperstack ship) loads too, but only files whose frontmatter sets `name`, so companion docs beside the profile stay out; a flat file wins a name clash. `superclaw agents` lists them |
| Repo map | `SUPERCLAW_REPO_MAP=0` turns it off; `superclaw repo-map [--json\|--query TEXT]` prints it | on; a deterministic map of the workspace (`git ls-files` when in a repository, else a walk that skips build and cache directories) rendered as counts, the files that usually matter, languages and paths, clamped to 6,000 bytes and placed in the system prompt as `<repo_map>` so the model knows the layout on turn one. `--query` ranks paths by term |
| Web search | `SUPERCLAW_SEARCH=google\|duckduckgo`, `GOOGLE_API_KEY`, `GOOGLE_CSE_ID` | `google` when both Google values are set, `duckduckgo` otherwise; DuckDuckGo needs nothing but rate-limits bursts, so requests are spaced 1.5 s apart and a refusal is retried once after 3 s before the error names the Google variables |
| Reader proxy | `SUPERCLAW_READER=https://r.jina.ai/` | off; when set, a page that answers with an HTTP error is fetched again as `<reader><url>` and the output says `Via:` |
| Skill names | `<plugin>:<skill>` for skills from a Claude-format plugin, `<dir>:<skill>` for a namespace directory under a skills root (a directory, or a symlink to one, holding skill directories, the layout hyperstack's installer creates with `~/.config/superclaw/skills/hyperstack -> <checkout>/skills`), bare names otherwise | the `skill` tool takes the full name and falls back to a bare name when exactly one skill matches, so `hyperstack:best-practices` and `best-practices` both load |
| Hyperstack | `superclaw plugin install --link ~/orkait/hyperstack`, then the MCP entry below in `~/.config/superclaw/mcp.json`; `superclaw doctor` and `superclaw mcp` confirm | hyperstack is three pieces and all three land: its 29 skills as `hyperstack:<name>`, its `hyper` and `website-builder` agents (`--agent hyper`), and its `SessionStart` bootstrap through the plugin hook (once per session). The MCP server is the same persistent container Claude Code uses, `{"mcpServers": {"hyperstack": {"command": "docker", "args": ["exec", "-i", "hyperstack-mcp", "bun", "/app/src/index.ts"], "env": {"HYPERSTACK_ROOT": "~/.hyperstack"}}}}` (118 tools as `mcp_hyperstack_*`); `node <checkout>/bin/hyperstack.mjs` works without Docker. Hyperstack's own installer (`bun run setup` in its checkout) knows superclaw as a platform: env `SUPERCLAW_PLUGIN_ROOT`, config `~/.config/superclaw/mcp.json`, skills `~/.config/superclaw/skills` |
| Claude Code config | `SUPERCLAW_CLAUDE_CONFIG=1`, honouring `CLAUDE_CONFIG_DIR` (default `~/.claude`) | off; when on, superclaw reads the same files Claude Code reads, so one setup serves both: skills from `~/.claude/skills` and `<workspace>/.claude/skills`, agents from `~/.claude/agents` and `.claude/agents`, commands from `~/.claude/commands` and `.claude/commands`, `CLAUDE.md` (or `.claude/CLAUDE.md`) as project guidelines where no `AGENTS.md` exists, every plugin in `~/.claude/plugins/installed_plugins.json`, MCP servers from `~/.claude.json` (user `mcpServers` plus the workspace's `projects` entry, minus `disabledMcpServers`) and, with `--trust-workspace`, the workspace `.mcp.json` minus `disabledMcpjsonServers`, and hooks from `~/.claude/settings.json` plus, with `--trust-workspace`, `.claude/settings.json` and `.claude/settings.local.json`. `superclaw doctor` prints `claude config on`. Own files still win: a superclaw skill, agent, command or server with the same name shadows the Claude one |
| Claude hook dialect | any hook written in Claude's nested `hooks` schema (settings files or a plugin's `hooks/hooks.json`) | `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PermissionRequest`, `Notification` (`permission_prompt`, `elicitation_dialog`), `PreCompact` (`trigger` auto or manual), `Stop`, `SubagentStop` and `SessionEnd` (`reason` `prompt_input_exit`, `clear`, `other`) map to superclaw's events; `PermissionRequest` may answer `hookSpecificOutput.decision.behavior: allow|deny` with a `message`, which settles the approval before any prompt. Tool matchers see Claude's tool names (`bash` is `Bash`, `read_file` `Read`, `write_file` `Write`, `edit_file` `Edit`, `glob` `Glob`, `grep` `Grep`, `web_fetch` `WebFetch`, `web_search` `WebSearch`, `skill` `Skill`, `ask_user` `AskUserQuestion`) and superclaw's, stdin carries `hook_event_name`, `session_id`, `cwd`, `source`, `tool_name` (Claude's name) and `tool_input` with `path` renamed `file_path`, the command runs through `/bin/sh -c` with `CLAUDE_PROJECT_DIR` set and, for a plugin hook, `CLAUDE_PLUGIN_ROOT`. Stdout may answer with `additionalContext`, `hookSpecificOutput.additionalContext`, `systemMessage`, `decision: block` plus `reason`, `hookSpecificOutput.permissionDecision: deny` plus `permissionDecisionReason` (blocks the tool with that reason), or `hookSpecificOutput.updatedInput` (replaces the tool's arguments, `file_path` mapped back to `path`), so the enforcement hooks a `~/.claude/settings.json` already carries run unchanged. `SessionStart` fires once per session per process with `source` `startup` or `resume` |
| Plugins | `superclaw plugin list\|install <dir\|git url\|plugin@marketplace>\|remove <id>` | none; a plugin is a directory with a `plugin.json` (`id`, `description`, `version`) that bundles any of `skills/`, `agents/`, `commands/`, `hooks.json` and `mcp.json`. User plugins live under `~/.config/superclaw/plugins`, project plugins under `<workspace>/.superclaw/plugins`; skills, agents and commands from a project plugin load always, its hooks and MCP servers only with `--trust-workspace`, the same rule as the bare workspace files |
| Marketplaces | `superclaw plugin marketplace add <owner/repo\|git url\|dir> [--link]`, `marketplace list`, `marketplace remove <name>`, then `superclaw plugin install <plugin>@<marketplace>` | none; a marketplace is a checkout with `.claude-plugin/marketplace.json` (`name`, `plugins[]` with `name` and `source`), the catalogue format Claude Code marketplaces publish, kept under `~/.config/superclaw/marketplaces/<name>`. Supported plugin sources: a relative path inside the checkout, `{"source": "url", "url": ...}`, `{"source": "github", "repo": "owner/repo"}` and `{"source": "git-subdir", "url": ..., "path": ..., "ref": ...}` (a branch, tag or commit). `superclaw plugin marketplace add orkait/hyperstack` then `superclaw plugin install hyperstack@hyperstack` is the no-checkout route to hyperstack |
| User commands | `/<name> args` in the TUI, `superclaw exec "/<name> args"` | none; `<workspace>/.superclaw/commands/<name>.md` then `~/.config/superclaw/commands/<name>.md`, frontmatter `description` `agent` `model` over a template. Every skill is also a command under its full name, the way Claude Code lists skills in its `/` menu. `superclaw commands` lists both, skills tagged `skill` |
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
| Transcript | `❯` user lines, assistant markdown, one card per tool call: status glyph (`◐` running, `✓`, `✗`), tool name, target (path, pattern, the tool's own one-line `description` when the command or code spans several lines, else its first line), then the body - a unified diff for `edit_file`/`write_file`, output lines for everything else, `§id` when stored. The head carries the outcome (`+a -b` for edits, `N lines` for everything else) and the call's seconds. Bodies are folded by default; diffs and failures show their first 12 lines. Click a card to unfold it, or press `ctrl+o` to unfold every card, current and future, and again to fold them. Child (`delegate`) calls are indented |
| Status bar | two rows at the bottom, nothing at the top. The first is where you are and what it costs, space separated: path, git branch, `used/window` with the fill percent, session cost, model, effort. The second is what the harness is doing, `·` separated and every segment conditional: mode with `shift+tab to cycle`, sandbox on or off, the cache hit share, tokens kept out of the window, the active agent, the session title. Once the context passes half the compaction trigger the first row switches from the fill percent to `N% until compaction`, because that is the number that predicts when the harness will act. The hints row shows only until the first message, so the steady state is two rows, not three |
| Working line | one row while a run is in flight, hidden otherwise: spinner, the real phase, then elapsed seconds (excluding time spent in a prompt), tool count and tokens so far. Phases: `recalling` while the graph is searched, `thinking` with the recall overview (`2 memories · 14 skills · repo map 40 files · 12 earlier messages`, counts only, never content), `writing` once text streams, the tool name with its target while a tool runs, `waiting for you` during a permission prompt, `cancelling` after `esc` |
| Title bar and status bar | workspace, branch and session id above; `● mode`, context reading `◔ 21.4K/1.0M · 2.1%`, run cost and tokens kept out below; the model sits on the composer border. Segments drop as the terminal narrows (tiers at 58, 80 and 100 columns) |
| Command palette | typing `/` lists matching commands with usage and help; `up`/`down` move the highlight, `tab` or enter on a highlighted row picks it, enter with nothing highlighted runs what was typed, `esc` closes it. A command that takes arguments is completed into the prompt; one without runs at once |

| Command | Effect |
|---|---|
| `/mode ask\|auto\|plan\|unsafe` | switch the permission mode for the session |
| `/model [list\|id]` | no argument opens the picker: recent models first, then one group per connected provider, type to filter, enter picks. `list` prints the same rows. An id switches at once, fuzzy when unique (`/model v4-pro`), and a model on a provider without a key opens setup for that provider. The choice is saved as `SUPERCLAW_MODEL` |
| `/effort low\|medium\|high\|off` | set the model's reasoning effort for the session, saved as `SUPERCLAW_EFFORT` and shown in the status bar |
| `/<name> [args]` | a user command from `.superclaw/commands`, or a skill (`/hyperstack:best-practices refactor the parser`): the template expands (`$ARGUMENTS`, `$1`..`$9` shell-split, `$$`) and runs as a prompt, a skill's body wrapped in `<skill>` with the request after it; the palette tags skills `skill:`; history keeps the typed line. Builtins win over user commands, user commands over skills, on a name collision |
| `/agent [name\|none]` | no argument lists the profiles and marks the active one; a name applies its prompt and narrows the tools; `none` clears it and restores the allowlist the command line asked for |
| `/attach <path>` | attach a file or image and leave its `[File #n]` or `[Image #n]` marker in the composer; the markers are the source of truth, as in Claude Code: when you send, the attachments whose markers are still in the text go with the message in marker order, and a marker you deleted takes its attachment out. A marker is one unit, like a chip in Claude Code: backspace, delete, ctrl+w, ctrl+u or ctrl+k touching any part of it removes the whole marker, and typing inside one lands after it, so a broken half-marker can never sit in the composer. The typed line, markers included, is what lands in history |
| drop, paste, `ctrl+v` | dropping a file onto the terminal pastes its path (plain, quoted or `file://`); superclaw attaches it and leaves `[Image #1]` or `[File #1]` in the composer. A paste of 3+ lines or over 2,000 chars becomes `[Pasted text #1 +N lines]` with the text attached whole instead of being cut to its first line. `ctrl+v` reads the clipboard the way Claude Code does: an image (`xclip` or `wl-paste` on Linux, `pngpaste` or `osascript` on macOS) is saved under `~/.local/share/superclaw/clipboard` and attached as `[Image #n]`, so a screenshot taken to the clipboard is one keypress away; text goes through the same paste rules |
| `/mcp` | the MCP tools that connected, the servers that were skipped and why, and any config problem |
| `/new` (aliases `/clear`, `/reset`), `/resume [id\|latest]`, `/fork [id\|latest]`, `/sessions [query]` | session lifecycle; `/new` starts a fresh session with an empty context and a zeroed status bar, as `/clear` does in Claude Code, and the old session stays resumable; bare `/resume` opens a picker over every session, grouped by project with the current one first, each row showing its title, age, git branch, stored size and model, with type to search and `ctrl+a` to reach the other projects, while `/resume <id>` and `/resume latest` still jump straight there; `/fork` copies a session (the current one by default) into a new one and continues it; `/sessions <query>` searches stored events by meaning instead of listing |
| `/usage` | calls, tokens, cached tokens and cost spent in this session; every model call is now stored as a `usage` event, so `superclaw usage` reports real totals for past sessions instead of zeros |
| Prompt cache | measured, not assumed: every model call records `cache_read_tokens`, the run summary ends with the share of the prompt served from cache (or says the provider reports none), and `superclaw doctor` totals it across recent sessions for the active model. On 2026-09-14 `openrouter/deepseek/deepseek-chat` returned 12,800 of 12,968 input tokens cached on the second turn of a run, while `opencode/deepseek-v4-flash-vision-exp` returned zero on the same prefix, which is worth roughly 30x on the input bill |
| `/context [prompt]` | what the next request costs, by category |
| `/facts [query\|as-of DATE [query]\|retract ID [reason]]` | facts learned from sources with their age and URL; `as-of` shows what was believed on a date |
| `/ask <question>` | answer from stored facts and results through the substrate's reader; prints the cited ids |
| `/maintain` | expire, decay stale facts, optimize, and show the store's health |
| `/snapshots`, `/rollback <name>` | snapshots taken in this process (one before the first unsafe run); rollback discards later writes and opens a fresh session |
| `/recall <§id\|query>` | bring back or search stored tool results, rendered as a card |
| `/compact` | summarize older turns into one message now, freeing the window before the next run; a session event, so it survives resume |
| `/retry` | run the last prompt again |
| `/rename <title>` | name the session; the title shows in the title bar and `/sessions` |
| `/export` | write the transcript to `superclaw-transcript-<id>.md` in the workspace |
| `/tools` | list every tool, its side effect, and whether it is hidden in the current mode |
| `/permissions` | show the mode, the session tool grants and the remembered command prefixes |
| `/doctor` | terminal, sandbox, mode, agent, model, store, MCP, the graph's embedder with node and edge counts, the store's health and last maintenance, the modern host tools present, and connected-provider health, read-only |
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

`superclaw mcp add docs -- npx -y some-mcp-server --env API_KEY=...` and `superclaw mcp add hosted --url https://mcp.example.com/mcp --header 'Authorization=Bearer ...'` write those entries; `superclaw mcp remove docs` drops one; `--scope project` targets the workspace file instead. `~/.config/superclaw/mcp.json` always, `<workspace>/.superclaw/mcp.json` only with `--trust-workspace`, because an entry there is a command this process runs. Servers connect concurrently under one deadline, so startup costs the slowest server rather than the sum, and a server that fails to start, times out or collides on a tool name is skipped with a reason instead of taking the launch down. Tools arrive as `mcp_<server>_<tool>`, deferred behind `tool_search`, gated as network side effects. A `command` entry is stdio; a `url` entry is streamable HTTP: JSON-RPC is POSTed with `Accept: application/json, text/event-stream`, the `Mcp-Session-Id` the server hands back on `initialize` is sent on every later request, and a response delivered as an SSE body is unwrapped to the message with the matching id. The legacy SSE transport (`"type": "sse"`) is refused with a reason, as is an entry that mixes the two shapes.

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
| One writer per store | supergraph takes an exclusive lock per path, so the first superclaw to open the brain owns it and serves it over a unix socket under `$XDG_RUNTIME_DIR/superclaw/<hash of the store path>.sock`; every later superclaw on the same store attaches and sends its queries there. Attaching waits up to 10 s for the owner's socket to answer a health query, because the lock is taken a moment before the socket exists and three sessions launched at once used to leave two of them dead with a misleading "restart that session". If the owner exits, the next query from an attached session takes the lock and starts serving. A superclaw older than this scheme holding the lock without a socket is reported, with the fix being to restart it |
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
