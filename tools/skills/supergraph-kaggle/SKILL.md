---
name: supergraph-kaggle
description: How to push, run, monitor, and debug supergraph benchmarks on Kaggle. Covers free-tier limits (max 2 concurrent runs), KGAT + legacy auth, validate-before-push protocol, model caching via Kaggle datasets, kernel_ctl.py control plane, and the top 5 error classes you will hit (module not found, circular import, tokenizer path, empty error message, token secrets). Use whenever you are about to push a Kaggle kernel, debug a failing run, or wire a new embedder into the cache.
compatibility: Requires supergraph repo checked out, kaggle CLI + kagglesdk installed, ~/.kaggle credentials provisioned.
metadata:
  author: orkait
  version: "1.0"
---

# SuperGraph on Kaggle

Everything you need to push, run, monitor, and debug supergraph benchmarks on Kaggle.

## When to use this skill

- About to push a Kaggle kernel (REQUIRED: ask permission first)
- Debugging a failing run
- Adding a new embedder or model that needs caching
- Hitting Kaggle auth / token / module errors

## Free tier limits + push protocol

**CRITICAL:** Free tier allows **max 2 concurrent kernel runs** at any time.

**ALWAYS ASK FOR PERMISSION BEFORE PUSHING TO KAGGLE.**

Before pushing:

1. **Ask user for permission** - never push without explicit approval
2. Check current run status: `python3 benchmarks/kaggle/kernel_ctl.py status`
3. Wait for previous run to complete (COMPLETE or ERROR status)
4. Do NOT push new versions while one is running
5. Validate locally first: `python3 benchmarks/kaggle/validate_before_push.py`
6. Each push increments kernel version; wasted pushes = wasted runs on free tier

## Authentication

Two auth methods. **KGAT is preferred** (new bearer token format).

### KGAT token (new)

```bash
echo "KGAT_xxxxxxxxxxxx" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Used automatically by `kagglesdk` (`kernel_ctl.py`).

### Legacy API key (fallback)

```bash
# ~/.kaggle/kaggle.json
{"username": "superkaiii", "key": "YOUR_API_KEY"}
chmod 600 ~/.kaggle/kaggle.json
```

Used by `kaggle` CLI (`kaggle kernels push`, `kaggle kernels status`).

Both can coexist. KGAT is used by `kernel_ctl.py`; legacy key is used by `kaggle` CLI.

## Validate before every push

**REQUIRED:** run validation locally before pushing. Catches errors before wasting GPU time and limited free-tier runs.

### Quick validation (imports + structure only)

```bash
python3 benchmarks/kaggle/validate_before_push.py --skip-run
```

### Full validation (includes 1-record benchmark)

```bash
python3 benchmarks/kaggle/validate_before_push.py
# (mini run is ON by default, takes ~2-3 min)
```

**What it checks:**

| Check | What |
|---|---|
| imports | supergraph, onnxruntime, huggingface_hub, docker_runner |
| circular | no circular import in supergraph package |
| models | jina-small + tinybert-ner structure valid |
| argv | all required flags present in sys.argv config |
| mini-run | 1 record through full pipeline (CPU, local models) |

**Exit codes:**

- `0` = all checks pass, safe to push
- `1` = failure detected, fix before pushing

## Model caching (skip downloads)

Models are uploaded as Kaggle datasets once and reused across runs. Saves ~5 min per run.

**Cached datasets:**

| Kaggle dataset | Model | Size |
|---|---|---|
| `superkaiii/tinybert-ner-onnx` | TinyBERT NER (entity extractor) | 15 MB |
| `superkaiii/jina-v5-small-onnx` | Jina v5 Small ONNX (embedder) | ~260 MB |

**How it works:**

- Datasets attached in `kernel-metadata.json` under `dataset_sources`
- Script checks `/kaggle/input/<slug>/` first. If found, symlinks it (no download)
- Falls back to HF download if not attached

**To add or update a cached model:**

```bash
# 1. Download model locally
.venv/bin/python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('owner/model-repo', local_dir='/tmp/my-model')
"

# 2. Init + upload as Kaggle dataset
kaggle datasets init -p /tmp/my-model
# edit /tmp/my-model/dataset-metadata.json (set id + title)
kaggle datasets create -p /tmp/my-model --dir-mode zip

# 3. Add slug to kernel-metadata.json dataset_sources
# 4. Add slug to EMBEDDER_KAGGLE_SLUG or NER_KAGGLE_SLUG in kaggle_benchmark.py CONFIG
```

## Push a kernel

```bash
# Push default kernel (supergraph-jina-v5-small)
kaggle kernels push -p benchmarks/kaggle/

# Output confirms pushed version:
# Kernel version 16 successfully pushed.
```

Kernel metadata files live in `benchmarks/kaggle/`:

| File | Kernel |
|---|---|
| `kernel-metadata.json` | `supergraph-jina-v5-small` (default, active) |

## Kernel control (`kernel_ctl.py`)

Full programmatic control via `kagglesdk` (KGAT auth, no CLI needed).

```bash
# Status
python3 benchmarks/kaggle/kernel_ctl.py status

# Logs (stdout + stderr)
python3 benchmarks/kaggle/kernel_ctl.py logs

# Cancel running kernel
python3 benchmarks/kaggle/kernel_ctl.py cancel

# Start kernel session
python3 benchmarks/kaggle/kernel_ctl.py run

# Different kernel
python3 benchmarks/kaggle/kernel_ctl.py logs --kernel supergraph-pipeline-refactored
```

## Monitor a run

```bash
# Check status
python3 benchmarks/kaggle/kernel_ctl.py status
# KernelWorkerStatus.RUNNING / ERROR / COMPLETE / CANCEL_REQUESTED

# Get logs (available after ~30-60s into run)
python3 benchmarks/kaggle/kernel_ctl.py logs 2>&1 | tail -50

# Filter for errors only
python3 benchmarks/kaggle/kernel_ctl.py logs 2>&1 | grep "\[ERR\]"

# Watch for key milestones
python3 benchmarks/kaggle/kernel_ctl.py logs 2>&1 | grep -E "system:|config:|evaluating|interrupted|PASS|FAIL|ERROR"
```

## Common errors + fixes

### `ModuleNotFoundError: No module named 'supergraph'`

Subprocess doesn't inherit `sys.path`. Fix: pass `PYTHONPATH` explicitly.

```python
env = os.environ.copy()
env["PYTHONPATH"] = "/kaggle/working/supergraph"
subprocess.check_call(
    [sys.executable, "scripts/some_script.py"],
    cwd="/kaggle/working/supergraph",
    env=env,
)
```

### `ImportError: cannot import name 'SuperGraph' from partially initialized module` (circular import)

Caused by absolute imports in `supergraph/supergraph/__init__.py`. Fix: switch to relative.

```python
from .store import SuperGraph           # relative - correct
from .core.store import CoreStore       # relative - correct
```

NOT:

```python
from supergraph.store import SuperGraph  # absolute - circular when on PYTHONPATH
```

### `FileNotFoundError: tokenizer.json not found in models/tinybert-ner`

Two causes:

1. TinyBERT not downloaded. Add to script:

   ```python
   snapshot_download(
       "onnx-community/TinyBERT-finetuned-NER-ONNX",
       local_dir="/kaggle/working/models/tinybert-ner",
       token=HF_TOKEN,
   )
   ```

2. Path mismatch. `--entity-model-dir` must point to the actual download location:

   ```python
   sys.argv = [..., "--entity-model-dir", "/kaggle/working/models/tinybert-ner", ...]
   ```

The extractor checks both `{model_dir}/tokenizer.json` and `{model_dir}/onnx/tokenizer.json`.

### `KernelWorkerStatus.ERROR` with empty failure message

Kaggle doesn't always populate `failure_message`. Get the real error from logs:

```bash
python3 benchmarks/kaggle/kernel_ctl.py logs 2>&1 | grep "\[ERR\]" | tail -20
```

### Push rejected / HF_TOKEN secrets issue

Never hardcode tokens in scripts. Read from Kaggle Secrets or private dataset.

```python
# Option 1: Private dataset (preferred)
token_file = "/kaggle/input/hf-token-private/hf_token.txt"
if os.path.exists(token_file):
    with open(token_file) as f:
        HF_TOKEN = f.read().strip()

# Option 2: Kaggle Secrets (UI)
from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
```

## Typical workflow

```bash
# 1. Make code changes

# 2. Validate locally
.venv/bin/python3 benchmarks/kaggle/validate_before_push.py

# 3. Commit + push branch
git add . && git commit -m "fix: ..."
git push origin refactor/simplify-retrieval-pipeline

# 4. Push kernel (clones latest branch at runtime)
kaggle kernels push -p benchmarks/kaggle/

# 5. Monitor
python3 benchmarks/kaggle/kernel_ctl.py status
python3 benchmarks/kaggle/kernel_ctl.py logs 2>&1 | tail -30

# 6. If failed, check logs, fix, repeat from step 1
python3 benchmarks/kaggle/kernel_ctl.py logs 2>&1 | grep "\[ERR\]"
```

## Key files

| File | Purpose |
|---|---|
| `benchmarks/kaggle/kaggle_benchmark.py` | Main benchmark runner (edit CONFIG block at top) |
| `benchmarks/kaggle/kernel_ctl.py` | kagglesdk kernel control |
| `benchmarks/kaggle/validate_before_push.py` | Pre-push local validator |
| `benchmarks/kaggle/kernel-metadata.json` | Kaggle kernel config (maps script to kernel) |
| `benchmarks/supergraph.json` | DSL tuning config (loaded via `SUPERGRAPH_CONFIG`) |
| `~/.kaggle/access_token` | KGAT bearer token for kagglesdk |
| `~/.kaggle/kaggle.json` | Legacy API key for kaggle CLI |
