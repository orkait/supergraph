# autoresearch

LLM-driven optimization ratchet for files under `supergraph/algos/`. Proposes
candidate implementations via an LLM, verifies them against the baseline for
correctness and speed, and promotes only those that measurably improve the
benchmark composite.

## What it is

A **completion-driven scientific ratchet**. Each iteration:

1. Pick a hypothesis (e.g., "vectorize", "algorithm", "scope_widen")
2. Build a prompt with the current baseline, bottleneck, and hypothesis constraint
3. Send one LLM call, extract Python source from the response
4. Run a correctness gate (isolated module comparison vs baseline)
5. Run a staleness gate (AST comparison vs recent attempts)
6. Run a tiered benchmark (quick filter → min-of-3 full bench)
7. If measurably faster than the stored best, promote to new baseline
8. Otherwise, consume 1 of the hypothesis's attempt quota and move on

## What it is not

This is **not** the full Karpathy autoresearch design. Karpathy's agent reads
code, profiles, plans, and iterates within a single experiment using a real
coding agent (Claude Code / Cursor / Aider). This system sends one shot to
`litellm.completion()` per iteration with a static prompt; the LLM has no
access to the repo, the bench results, or prior candidates. Consequently:

- It finds **local optimizations** the LLM can guess from baseline + constraint
- It **does not explore** alternative data layouts or upstream refactors
- It **does not profile** the actual hot path; it trusts the composite metric

For a task like "replace a Python for-loop with `np.fromiter`", this works
well. For "restructure the edge representation throughout the callers",
this is the wrong tool.

## Files

```
autoresearch/
  run_loop.py        Main loop: scientific ratchet with budget, gates, bench
  correctness.py     Isolated module comparison gate (pre-bench)
  program.md         LLM system prompt (allowed imports, output format)
  config.example.json  Sanitized config template
  config.json        Real config with API keys (gitignored, copy from example)
  README.md          This file
```

Runtime artifacts live in `/.autoresearch/` (gitignored):

```
.autoresearch/
  <algo>_checkpoint.json    Per-algo ratchet state (baseline, best_score, attempts, history)
  <algo>_results.jsonl      Per-iteration result log (JSONL)
  candidates/<algo>/        Every LLM response saved for post-hoc inspection
```

## Core design principles

### 1. The ratchet

`best_score` is **monotonic non-increasing**. It can only decrease on a
confirmed win or be tightened by a drift-refresh re-measurement. It cannot
inflate due to measurement noise. This is load-bearing: without monotonicity,
noisy re-measurements can artificially raise the baseline, making regressed
candidates look like wins.

```python
# Drift refresh - re-measure current baseline_code
if fresh_score < best_score:
    best_score = fresh_score          # tighter measurement - trust it
    baseline_result = fresh
else:
    # Fresh is worse - noise, keep stored best
    pass
```

### 2. Correctness gate

The benchmark tests in `benchmarks/algos/test_*_bench.py` call functions but
**do not assert on return values**. Without the correctness gate, a candidate
that returns `None` or a garbage output would pass the bench as long as it
runs fast enough. The gate loads baseline and candidate into isolated module
namespaces via `exec` into fresh `types.ModuleType` objects, runs each on a
panel of deterministic inputs that mirror the benchmark fixtures, and compares
outputs deeply (numpy/scipy-sparse/dict/list/set aware, `np.allclose` with
`rtol=1e-5, atol=1e-8` for floats).

Per-algo input generators live in `correctness.py::_DISPATCH`. Adding a new
algo requires adding an entry with the functions to test and their args.

### 3. Single-counter hypothesis budget

Each hypothesis gets `attempts_per_hypothesis` (default 3) attempts per
model per baseline. Any non-win outcome counts as 1 attempt:

| Outcome | Effect |
|---|---|
| `purity_rejected` | attempts += 1 |
| `correctness_rejected` | attempts += 1 |
| `stale` | attempts += 1 |
| `slow_reject_quick` (>5% worse on quick bench) | attempts += 1 |
| `slow_reject_full` (not faster on full bench) | attempts += 1 |
| `bench_failed` (pytest crash) | attempts += 1 |
| `timeout` (SIGALRM fired) | attempts += 1 |
| `error` (network/API exception) | attempts += 1 |
| **`win`** (faster, correct) | **reset all attempts**, promote baseline |

When `attempts[h] >= attempts_per_hypothesis`, the hypothesis is locked out
for the current baseline. `scope_widen` is held back until all standard
hypotheses are exhausted. When all hypotheses are locked, the loop
terminates with "locally optimal".

Max LLM calls per `(algo, model)` = `6 hypotheses × attempts_per_hypothesis`
= **18** at the default setting.

### 4. Tiered bench with min-based measurement

`pytest-benchmark` reports `min`, `median`, `mean`, `stddev`. For latency
microbenchmarks, **min is the correct statistic** - background noise (GC,
context switches, thermal, CPU contention) only slows runs down, never
speeds them up. `bench_one.py` was updated to report `min_us` instead of
`mean_us`.

The tiered bench does:

1. **Quick bench** (1x, fewer samples) to filter obvious losers - reject if
   quick_score > `best_score * noise_tolerance` (default `1.02`)
2. **Full bench** (min of `full_bench_repeats`, default 3) to confirm anything
   plausible

The `min-of-N` across full bench repeats filters transient noise from
parallel loops or scheduler jitter.

### 5. IterationTimeout propagation

`IterationTimeout` inherits from `BaseException`, **not** `Exception`. This
is deliberate: libraries like litellm have internal `except Exception`
wrappers that would otherwise swallow our SIGALRM-raised timeout and let
the LLM call retry indefinitely. By using `BaseException`, the signal
propagates cleanly through all third-party error handlers until it reaches
the iteration-level handler that was designed to catch it.

```python
class IterationTimeout(BaseException):  # NOT Exception
    pass
```

### 6. Baseline is sacred

During bench, the candidate is written to `algo_path(algo)` and the baseline
is **always restored in a `finally` block**, even if the bench crashes.
Promotion to a new baseline only happens on a confirmed win, after bench
completion. At no point between iterations does the `.py` file contain an
unverified candidate.

## Configuration model

Two-level hierarchy: **providers** own connection settings, **models** are
leaves under them. A single API key per provider is reused by all its models.

Secrets never live in `config.json`. Each provider declares an `env_key`
that points at a field on the typed `ENV` object in `/env.py`; values
come from `/.env` (gitignored) or the shell. See `/.env.example` for the
full list of variables.

```json
{
  "active_provider": "local_ollama",
  "active_model": "qwen3-coder-next:cloud",
  "provider_fallback_order": ["local_ollama", "openrouter"],
  "providers": {
    "local_ollama": {
      "base_url": "http://localhost:11434",
      "env_key": "ollama_key",
      "is_local": true,
      "litellm_prefix": "ollama_chat",
      "models": {
        "qwen3-coder-next:cloud": { "notes": "..." },
        "minimax-m2.7:cloud":     { "notes": "..." }
      },
      "model_fallback_order": [
        "qwen3-coder-next:cloud",
        "minimax-m2.7:cloud"
      ]
    },
    "openrouter": {
      "base_url": "https://openrouter.ai/api/v1",
      "env_key": "openrouter_key",
      "is_local": false,
      "litellm_prefix": "openrouter",
      "models": {
        "qwen/qwen3-coder-next:nitro": { "notes": "..." },
        "z-ai/glm-5.1:nitro":          { "notes": "..." }
      },
      "model_fallback_order": ["qwen/qwen3-coder-next:nitro"]
    }
  }
}
```

**Fallback order of operations** in `get_llm_proposal`:

1. Try `active_provider` + `active_model`
2. If that model fails (error, empty response, no def), try
   `model_fallback_order` on the same provider
3. If all models on that provider fail, move to the next entry in
   `provider_fallback_order` and repeat with ITS model order
4. If all providers and models fail, raise `RuntimeError`

Legacy flat schema (one provider per model, no `models` dict) is
auto-migrated in `migrate_config()` at load time.

## Usage

### Setup

```bash
# Copy the shape-only config template (config.json is gitignored):
cp tools/autoresearch/config.example.json tools/autoresearch/config.json

# Put real API keys in /.env at the repo root. config.json never holds
# secrets; it only declares env_key pointers.
cp .env.example .env
# edit /.env and set OPENROUTER_API_KEY + OLLAMA_API_KEY

# Run a single loop
python -m tools.autoresearch.run_loop --algo spreading --iterations 18
```

### CLI flags

```
--algo ALGO          Target algo file (graph, compact, fusion, spreading, eviction, text)
--iterations N       Run up to N more iterations from wherever we resume (relative, not absolute)
--model MODEL        Override active_model; auto-picks the provider that owns this model
--provider PROVIDER  Override active_provider (usually unnecessary when --model is given)
```

### Parallel multi-model sweep

Iterations are network-bound (LLM calls), so 4x parallelism is effectively
free. Each loop has its own checkpoint + candidate directory, no file
collisions. Run different algos with different models:

```bash
python -m tools.autoresearch.run_loop --algo graph     --iterations 18 --model qwen/qwen3-coder-next:nitro &
python -m tools.autoresearch.run_loop --algo fusion    --iterations 18 --model minimax/minimax-m2.7:nitro &
python -m tools.autoresearch.run_loop --algo spreading --iterations 18 --model qwen3-coder-next:cloud &
python -m tools.autoresearch.run_loop --algo compact   --iterations 18 --model z-ai/glm-5.1:nitro &
```

### Mid-run config edits

`config.json` is re-read on every iteration. You can edit `active_model`,
`noise_tolerance`, `attempts_per_hypothesis`, or `max_iterations` while a
loop is running and the changes take effect on the next iteration. CLI
overrides (`--model`, `--provider`) are sticky - they are re-applied after
every config reload.

## Retroactive verification (critical)

**Never trust a win at face value.** The loop's `best_score` is measured
against the *stored baseline*, which may have been inflated by noise. After
a loop terminates (or when you're ready to commit a winner), re-measure
the candidate against git HEAD independently:

```python
import subprocess, json, math
from tools.autoresearch.correctness import check_correctness

original = subprocess.check_output(['git', 'show', f'HEAD:supergraph/algos/{algo}.py'], text=True)
winner = open(f'supergraph/algos/{algo}.py').read()

# Correctness
err = check_correctness(winner, original, algo)
assert err is None, f"correctness failed: {err}"

# Performance (against git HEAD, with stash-swap to compare)
# - bench winner (current .py)
# - git stash  → bench baseline  → git stash pop
# - compare composites (geometric mean of min_us values)
```

Multiple winners in the development of this tool were discovered to be
false positives only via this retroactive check. Do not skip it.

## Hypothesis taxonomy

Defined in `HYPOTHESIS_TYPES` in `run_loop.py`. Each has a `name` and a
`constraint` that gets injected into the prompt:

| Name | Constraint summary |
|---|---|
| `vectorize` | Eliminate Python for/while loops; use numpy ufuncs, np.where, np.nonzero |
| `algorithm` | Use a fundamentally different algorithm; must use at least one of np.searchsorted / argsort / unique / bincount / scipy.csgraph |
| `memory` | Pre-allocate all output arrays; no list.append, no concatenate in loops |
| `dtype` | Use smallest correct dtype (int32, float32); minimise casting |
| `simd` | Maximise SIMD throughput via simsimd or contiguous int32/float32 arrays |
| `scope_widen` | May modify helper functions and internal data structures (public signatures preserved) |

`scope_widen` is unlocked only after all standard hypotheses are exhausted.
The staleness gate uses **whole-file AST comparison** for `scope_widen`
iterations (since the target function may stay identical while helpers
change), and **target-function-only AST comparison** for the others.

## Extending

### Add a new algo

1. Create `supergraph/algos/<algo>.py` with the target functions
2. Create `benchmarks/algos/test_<algo>_bench.py` with pytest-benchmark tests
3. Register the algo in `benchmarks/algos/bench_one.py::ALGO_TO_FILE` and
   `ALGO_TO_BENCH`
4. Add an input generator in `autoresearch/correctness.py::_DISPATCH` for
   the algo - follow the pattern of the existing entries, using small
   deterministic inputs that exercise the public functions
5. Run `python -m tools.autoresearch.run_loop --algo <name> --iterations 18 --model <model>`

### Add a new hypothesis

1. Append an entry to `HYPOTHESIS_TYPES` in `run_loop.py`
2. Write a clear, LLM-actionable `constraint` string
3. Decide if it needs whole-file staleness comparison (like `scope_widen`)
   or target-function-only (default)

### Add a new model

1. Add an entry under `providers.<provider>.models` in `config.json`
2. Optionally add it to `model_fallback_order` (remove it later if it's
   slow or produces poor candidates)
3. Test it: `python -m tools.autoresearch.run_loop --algo X --iterations 1 --model <model>`

If the model is on a new provider (e.g., anthropic), add a new top-level
provider entry with its `base_url`, `env_key`, and `litellm_prefix`.
Declare the matching field on `_Env` in `/env.py` and add the raw env
var to `/.env.example`.

## Known limitations

- **Correctness gate is not exhaustive.** It tests 3-6 input cases per algo.
  Edge cases (empty inputs, max sizes, special values) may slip through.
  The existing bench tests have no assertions and cannot be relied upon.
- **Baseline is the oracle.** If the original `.py` in git HEAD has a bug,
  the correctness gate will happily accept candidates that reproduce it.
- **Parallel loops create measurement contention.** Initial baseline
  measurements taken while 4 loops are spawning can be inflated; the
  monotonic `best_score` guards against this downstream but the very
  first measurement is still the oracle.
- **Not an agent.** The LLM has no read access to the repo and cannot
  profile, run tests, or iterate within a single experiment. Each
  iteration is a single completion call. Wins come from patterns the
  model can guess from baseline + constraint.
- **Per-algo correctness inputs are hand-written.** There is no automatic
  discovery of function signatures or test fixtures; adding a new algo
  requires mirroring the relevant `conftest.py` fixtures in
  `correctness.py::_DISPATCH`.

## Historical bugs to be aware of

These have been fixed but are worth understanding before modifying the loop:

- **Drift refresh used to overwrite `best_score` unconditionally**, allowing
  noise-inflated re-measurements to raise the baseline and produce false
  wins. Fixed by making drift refresh only tighten `best_score`.
- **`IterationTimeout` used to inherit from `Exception`**, so litellm's
  internal `except Exception` would swallow it and retry slow LLM calls
  forever. Fixed by inheriting from `BaseException`.
- **`bench_one.py` used to report `mean_us`**, which is sensitive to
  background noise and hides small real wins while surfacing noise as
  "improvements". Fixed by switching to `min_us`.
- **Two counters (`slow_reject_budget` and `max_attempt_retries`) made
  the budget model confusing and inconsistent.** Replaced with a single
  `attempts_per_hypothesis` counter where any non-win outcome decrements.

## When to update this README

- New design invariants (not just bug fixes)
- New file or module in `autoresearch/`
- Changes to the config schema
- Changes to the hypothesis taxonomy
- New known limitations worth warning future readers about
