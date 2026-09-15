# superclaw on Terminal-Bench

Harbor agent adapter that installs superclaw inside a task container and runs it headless.

## Use

```bash
uv tool install harbor
uv pip install -e tools/terminal-bench
harbor run -d terminal-bench@2.0 \
  --agent superclaw_terminal_bench:Superclaw \
  -m openrouter/deepseek/deepseek-chat-v3.1 \
  -n 2
```

`--install-only` stops after the container install, which is the cheapest way to check the install script.

The adapter forwards provider keys from the host environment and pins the brain to `/tmp/superclaw-brain` so each trial starts clean. Token counts and cost come from the `usage` events in superclaw's `stream-json` output.
