"""
Algo Autoresearch - Scientific ratchet loop for supergraph/algos/.

Design principles:
  - Baseline is sacred: never modified until a statistically confirmed winner
  - Clean room: LLM sees proven best + bottleneck + hard constraint. No failure history.
  - Tiered bench: quick filter → full confirm (avoids wasting time on clear losers)
  - AST-level staleness: detect structural repetition on the target function only
  - Hypothesis exhaustion: track what was tried per baseline, unlock wider scope as fallback
  - Full checkpoint: restart continues exactly where it left off

Usage:
    python -m tools.autoresearch.run_loop
    python -m tools.autoresearch.run_loop --algo compact
    python -m tools.autoresearch.run_loop --algo graph --iterations 30
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ALGO_DIR = REPO_ROOT / "src" / "supergraph" / "algos"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
from tools.autoresearch.providers import CONFIG_PATH as CONFIG_FILE
PROGRAM_FILE = Path(__file__).resolve().parent / "program.md"

_current_algo: str = ""
_checkpoint: dict = {}
_config_cache: dict | None = None
_config_mtime: float = 0


# ---------------------------------------------------------------------------
# Hypothesis taxonomy - hard API-level constraints, not labels
# ---------------------------------------------------------------------------

HYPOTHESIS_TYPES: list[dict] = [
    {
        "name": "vectorize",
        "constraint": (
            "Eliminate ALL Python for/while loops from the target function. "
            "Use numpy ufuncs, np.where, np.nonzero, boolean indexing only. "
            "No list comprehensions in the hot path."
        ),
    },
    {
        "name": "algorithm",
        "constraint": (
            "Use a fundamentally different algorithm. "
            "You MUST use at least one of: np.searchsorted, np.argsort, np.unique with "
            "return_inverse, np.bincount, or scipy.sparse.csgraph primitives."
        ),
    },
    {
        "name": "memory",
        "constraint": (
            "Pre-allocate ALL output arrays before any computation. "
            "No list.append(), no intermediate list construction, no np.concatenate in loops. "
            "Write results directly into pre-allocated buffers."
        ),
    },
    {
        "name": "dtype",
        "constraint": (
            "Force all arrays to the smallest correct dtype (int32 for indices, float32 for weights). "
            "Minimise dtype casting in hot paths. "
            "Use np.empty instead of np.zeros where safe."
        ),
    },
    {
        "name": "simd",
        "constraint": (
            "Maximise SIMD throughput. Use simsimd if applicable, or ensure all array ops use "
            "contiguous int32/float32 arrays. Avoid Python scalars in the hot path. "
            "Use np.ascontiguousarray where needed."
        ),
    },
    {
        "name": "scope_widen",
        "constraint": (
            "You MAY modify helper functions or data structures to enable a faster target function. "
            "You MAY change internal representations. "
            "All public function SIGNATURES must remain identical to the baseline. "
            "Cross-function optimisation is allowed and encouraged."
        ),
    },
]


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

class IterationTimeout(BaseException):
    """Abort-the-iteration signal.

    Inherits from BaseException (NOT Exception) so that library code like
    litellm's `except Exception:` wrappers cannot swallow it. The SIGALRM
    budget must propagate all the way up to the iteration-level handler,
    otherwise the retry loop inside get_llm_proposal can extend LLM calls
    indefinitely past the configured iteration_timeout.
    """
    pass


def _timeout_handler(signum, frame):
    raise IterationTimeout("Iteration exceeded configured timeout")


def _graceful_shutdown(signum, frame):
    name = signal.Signals(signum).name
    print(f"\n\nReceived {name}. Saving checkpoint...")
    save_checkpoint(_current_algo)
    print("Done.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def migrate_config(cfg: dict) -> dict:
    """Convert old flat provider schema → nested providers/models schema.

    Old: providers[<pid>] = {base_url, model, api_key}  # one model per provider
    New: providers[<pid>] = {base_url, api_key, models: {<name>: {...}}, model_fallback_order: [...]}
         + active_model at top level
    """
    providers = cfg.get("providers", {})
    if not providers:
        return cfg
    is_old = any(
        ("model" in p and "models" not in p)
        for p in providers.values()
    )
    if not is_old:
        return cfg

    print("  Migrating config: flat provider schema → nested schema")

    buckets: dict = {}
    legacy_to_model: dict = {}
    for pid, p in providers.items():
        key = (p.get("base_url", ""), p.get("api_key", ""))
        bucket = buckets.setdefault(key, {
            "base_url": p.get("base_url", ""),
            "api_key": p.get("api_key", ""),
            "is_local": ("localhost" in p.get("base_url", "")
                         or "127.0.0.1" in p.get("base_url", "")),
            "models": {},
            "model_fallback_order": [],
        })
        model_name = p.get("model", "")
        if model_name:
            bucket["models"][model_name] = {"notes": p.get("notes", "")}
            if model_name not in bucket["model_fallback_order"]:
                bucket["model_fallback_order"].append(model_name)
            legacy_to_model[pid] = model_name

    new_providers: dict = {}
    provider_order: list = []
    for bucket in buckets.values():
        name = "local_ollama" if bucket["is_local"] else f"provider_{len(new_providers)}"
        new_providers[name] = bucket
        provider_order.append(name)

    legacy_active = cfg.get("active_provider", "")
    active_model = legacy_to_model.get(legacy_active, "")
    active_provider = provider_order[0] if provider_order else ""
    for pname, pbucket in new_providers.items():
        if active_model in pbucket["models"]:
            active_provider = pname
            break
    if not active_model and active_provider:
        order = new_providers[active_provider].get("model_fallback_order", [])
        if order:
            active_model = order[0]

    cfg["providers"] = new_providers
    cfg["active_provider"] = active_provider
    cfg["active_model"] = active_model
    cfg["provider_fallback_order"] = provider_order
    return cfg


def load_config() -> dict:
    global _config_cache, _config_mtime
    try:
        mtime = CONFIG_FILE.stat().st_mtime
        if mtime > _config_mtime:
            raw = json.loads(CONFIG_FILE.read_text())
            _config_cache = migrate_config(raw)
            _config_mtime = mtime
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  Config load failed: {e} - using cached")
    return _config_cache  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def algo_path(algo: str) -> Path:
    return ALGO_DIR / f"{algo}.py"


def baseline_path(algo: str) -> Path:
    return ALGO_DIR / f"{algo}.py.baseline"


def checkpoint_path(algo: str) -> Path:
    d = REPO_ROOT / ".autoresearch"
    d.mkdir(exist_ok=True)
    return d / f"{algo}_checkpoint.json"


def log_path(algo: str) -> Path:
    d = REPO_ROOT / ".autoresearch"
    d.mkdir(exist_ok=True)
    return d / f"{algo}_results.jsonl"


def candidate_dir(algo: str) -> Path:
    d = REPO_ROOT / ".autoresearch" / "candidates" / algo
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_candidate(algo: str, iteration: int, hypothesis: str, model: str,
                   raw_response: str, extracted_code: str):
    """Persist raw LLM output + extracted code for post-hoc inspection."""
    safe_model = model.replace("/", "_").replace(":", "_")
    base = candidate_dir(algo) / f"iter_{iteration:04d}_{hypothesis}_{safe_model}"
    base.with_suffix(".raw.txt").write_text(raw_response)
    base.with_suffix(".py").write_text(extracted_code)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(algo: str):
    if not algo:
        return
    checkpoint_path(algo).write_text(json.dumps(_checkpoint, indent=2, default=str))


def load_checkpoint(algo: str) -> bool:
    global _checkpoint
    cp = checkpoint_path(algo)
    if not cp.exists():
        return False
    try:
        _checkpoint = json.loads(cp.read_text())
        if _checkpoint.get("baseline_code"):
            baseline_path(algo).write_text(_checkpoint["baseline_code"])
            algo_path(algo).write_text(_checkpoint["baseline_code"])
        return True
    except Exception as e:
        print(f"  Checkpoint load failed: {e} - starting fresh")
        return False


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def _python() -> str:
    return str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def run_bench(algo: str, fast: bool) -> dict | None:
    cmd = [_python(), "-m", "benchmarks.algos.bench_one", algo, "--quiet", "--json"]
    if fast:
        cmd.append("--fast")
    try:
        result = subprocess.run(
            cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        print("  Bench timed out")
        return None
    if result.returncode != 0:
        print(f"  Bench failed (exit {result.returncode}): {result.stderr[:200]}")
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  Bench output unparseable: {result.stdout[:200]}")
        return None


def composite_score(bench_result: dict) -> float:
    values = list(bench_result["metrics"].values())
    if not values:
        return float("inf")
    return math.exp(sum(math.log(max(v, 1e-9)) for v in values) / len(values))


def identify_bottleneck(bench_result: dict) -> str:
    return max(bench_result["metrics"], key=lambda k: bench_result["metrics"][k])


def test_metric_to_fn_name(test_metric: str, source_code: str) -> str:
    """Map 'TestApplySlotRemap::test_100k_edges' → 'apply_slot_remap_to_edges'."""
    import re as _re
    class_name = test_metric.split("::")[0]
    bare = _re.sub(r"^Test", "", class_name)
    snake = _re.sub(r"([A-Z])", r"_\1", bare).lstrip("_").lower()
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return snake
    fn_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if snake in fn_names:
        return snake
    matches = [f for f in fn_names if f.startswith(snake)]
    return matches[0] if matches else snake


def run_bench_repeated(algo: str, fast: bool, repeats: int) -> dict | None:
    """Run bench N times, return the run with the minimum composite score.

    Taking the MIN across repeats filters transient noise (GC, context switches,
    parallel loop CPU contention). Min is the correct statistic for latency
    microbenchmarks - noise only makes runs slower, never faster.
    """
    best: dict | None = None
    best_score = float("inf")
    for _ in range(max(1, repeats)):
        r = run_bench(algo, fast=fast)
        if r is None:
            continue
        s = composite_score(r)
        if s < best_score:
            best_score = s
            best = r
    return best


def run_bench_tiered(
    algo: str,
    best_score: float,
    noise_tolerance: float = 1.05,
    full_repeats: int = 3,
) -> tuple[dict | None, str]:
    """
    Quick bench filters out *clearly worse* candidates only.

    noise_tolerance = 1.05 means: reject only if quick_score is >5% worse than best.
    Anything within noise (or any improvement, however small) proceeds to full bench.

    Full bench runs `full_repeats` times; the run with the MIN composite is used
    as the authoritative measurement (filters transient noise from parallel loops).

    Returns (result, tier) where tier in: "rejected", "full", "failed".
    """
    quick = run_bench(algo, fast=True)
    if quick is None:
        return None, "failed"

    quick_score = composite_score(quick)
    if quick_score > best_score * noise_tolerance:
        return quick, "rejected"

    print(f"  Quick bench plausible ({quick_score:.2f} us vs best {best_score:.2f} us) "
          f"- running full bench (min of {full_repeats})...")
    full = run_bench_repeated(algo, fast=False, repeats=full_repeats)
    if full is None:
        return quick, "failed"

    return full, "full"


# ---------------------------------------------------------------------------
# Purity gate
# ---------------------------------------------------------------------------

def check_purity(code: str) -> str | None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from benchmarks.algos.allowlist import allowed_import_names, is_forbidden
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    allowed = allowed_import_names()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                m = alias.name
                if is_forbidden(m):
                    return f"Forbidden import: {m!r}"
                if m.split(".")[0] not in allowed:
                    return f"Not-allowlisted import: {m!r}"
        elif isinstance(node, ast.ImportFrom):
            m = node.module or ""
            if is_forbidden(m):
                return f"Forbidden import: {m!r}"
            if m and m.split(".")[0] not in allowed:
                return f"Not-allowlisted import: {m!r}"
    return None


# ---------------------------------------------------------------------------
# Staleness detection - AST of target function only
# ---------------------------------------------------------------------------

def fn_ast_dump(code: str, fn_name: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return ast.dump(node, annotate_fields=False)
    return ""


def file_ast_dump(code: str) -> str:
    try:
        return ast.dump(ast.parse(code), annotate_fields=False)
    except SyntaxError:
        return ""


def is_stale(candidate: str, baseline: str, recent_asts: list[str], fn: str, whole_file: bool = False) -> bool:
    if whole_file:
        cand_ast = file_ast_dump(candidate)
        if not cand_ast:
            return True
        if cand_ast == file_ast_dump(baseline):
            return True
        return cand_ast in recent_asts
    cand_ast = fn_ast_dump(candidate, fn)
    if not cand_ast:
        return True
    if cand_ast == fn_ast_dump(baseline, fn):
        return True
    return cand_ast in recent_asts


# ---------------------------------------------------------------------------
# Hypothesis management
# ---------------------------------------------------------------------------

def next_hypothesis(attempts: dict, limit: int, allow_widen: bool) -> dict | None:
    """Pick first hypothesis that hasn't used up its attempt quota.

    Every iteration on a hypothesis counts as 1 attempt, regardless of outcome
    (win, slow reject, purity fail, stale, timeout, correctness drift, error).
    When attempts[h] >= limit, the hypothesis is locked out for the current
    baseline. A WIN resets all attempts to 0 (new baseline gets fresh budget).
    """
    for h in HYPOTHESIS_TYPES:
        name = h["name"]
        if name == "scope_widen" and not allow_widen:
            continue
        if attempts.get(name, 0) >= limit:
            continue
        return h
    return None


def fresh_attempts() -> dict:
    return {h["name"]: 0 for h in HYPOTHESIS_TYPES}


# ---------------------------------------------------------------------------
# LLM - clean room prompt
# ---------------------------------------------------------------------------

def get_env_manifest() -> str:
    try:
        r = subprocess.run(
            [_python(), "-m", "benchmarks.algos.dump_env", "--compact"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            out_lines = r.stdout.splitlines()
            trimmed = []
            skip = False
            for l in out_lines:
                if l.startswith("# forbidden:"):
                    skip = True
                if not skip:
                    trimmed.append(l)
            return "\n".join(trimmed).strip()
    except Exception:
        pass
    return "numpy, scipy available"


def build_prompt(
    baseline_code: str,
    bottleneck_fn: str,
    bottleneck_us: float,
    total_us: float,
    hypothesis: dict,
    algo: str,
    env_manifest: str,
    program_text: str,
) -> str:
    pct = bottleneck_us / max(total_us, 1e-9) * 100
    return f"""{program_text}

---

## Environment
{env_manifest}

## Proven baseline - supergraph/algos/{algo}.py
```python
{baseline_code}
```

## Bottleneck
Function `{bottleneck_fn}` costs {bottleneck_us:.0f} us - {pct:.0f}% of total latency.
All other functions are fast. Focus exclusively on `{bottleneck_fn}`.

## Hypothesis: {hypothesis['name']}
Hard constraint: {hypothesis['constraint']}

Optimise ONLY `{bottleneck_fn}` using the constraint above.
Keep all other functions structurally identical to the baseline.
Return the complete file. No markdown. No explanation.
"""


def get_llm_proposal(prompt: str, config: dict) -> tuple[str, str, str]:
    """Call LLM with provider → model fallback chain.

    Order tried: active_provider/active_model → model_fallback_order on same provider
                 → provider_fallback_order with each provider's models.

    Returns (extracted_code, model_used, raw_response).
    """
    from supergraph.llm_runner import LLMRunner
    from tools.autoresearch.providers import resolve_providers

    runner = LLMRunner(resolve_providers(config), timeout_s=800)
    raw_response, model = runner.call_sync_verbose(prompt, max_tokens=4096, temperature=0.7)

    if not raw_response.strip():
        raise RuntimeError("All providers/models failed: empty response")

    # Extract code: markdown fences, then trim leading prose.
    new_code = raw_response
    if "```python" in new_code:
        new_code = new_code.split("```python")[1].split("```")[0].strip()
    elif "```" in new_code:
        new_code = new_code.split("```")[1].split("```")[0].strip()

    lines = new_code.split("\n")
    start = 0
    for idx, ln in enumerate(lines):
        s = ln.lstrip()
        if s.startswith(("from ", "import ", "def ", "class ", "@", "#!", '"""', "'''", "# ")):
            start = idx
            break
    new_code = "\n".join(lines[start:]).strip()

    if "def " not in new_code:
        raise RuntimeError(f"No function defs in LLM response from {model!r}")

    print(f"  Got {len(new_code)} chars from {model} (raw: {len(raw_response)})")
    return new_code, model, raw_response


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_result(algo: str, entry: dict):
    with open(log_path(algo), "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _apply_overrides(config: dict, model_override: str | None, provider_override: str | None) -> dict:
    """Inject CLI overrides onto the loaded config (in-memory, non-persistent)."""
    if provider_override:
        config["active_provider"] = provider_override
    if model_override:
        config["active_model"] = model_override
        # Auto-pick the provider that owns this model if provider wasn't also overridden
        if not provider_override:
            for pid, p in config.get("providers", {}).items():
                if model_override in p.get("models", {}):
                    config["active_provider"] = pid
                    break
    return config


def _validate_overrides(config: dict, model_override: str | None, provider_override: str | None):
    """Fail fast if overrides point at unknown provider/model."""
    providers = config.get("providers", {})
    if provider_override and provider_override not in providers:
        available = list(providers.keys())
        print(f"ERROR: --provider '{provider_override}' not found. Available: {available}")
        sys.exit(1)
    if model_override:
        all_models: list = []
        for p in providers.values():
            all_models.extend(p.get("models", {}).keys())
        if model_override not in all_models:
            print(f"ERROR: --model '{model_override}' not found. Available: {sorted(set(all_models))}")
            sys.exit(1)


def run_loop(
    algo_override: str | None = None,
    iterations_override: int | None = None,
    model_override: str | None = None,
    provider_override: str | None = None,
):
    global _current_algo, _checkpoint

    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    config = load_config()
    _validate_overrides(config, model_override, provider_override)
    config = _apply_overrides(config, model_override, provider_override)

    algo = algo_override or config.get("algo", "compact")
    _current_algo = algo

    print(f"=== Algo Autoresearch (scientific) ===")
    print(f"Algo:     {algo}")
    print(f"Provider: {config.get('active_provider')}"
          + (" (--provider)" if provider_override else ""))
    print(f"Model:    {config.get('active_model')}"
          + (" (--model)" if model_override else ""))
    print()

    if not algo_path(algo).exists():
        print(f"ERROR: {algo_path(algo)} not found")
        sys.exit(1)

    env_manifest = get_env_manifest()
    program_text = PROGRAM_FILE.read_text() if PROGRAM_FILE.exists() else ""

    # --- Restore or initialise checkpoint ---
    restored = load_checkpoint(algo)
    if restored and _checkpoint.get("baseline_code"):
        baseline_code = _checkpoint["baseline_code"]
        baseline_result = _checkpoint["baseline_result"]
        best_score = _checkpoint["best_score"]
        attempts = _checkpoint.get("attempts") or fresh_attempts()
        allow_widen = _checkpoint.get("allow_scope_widen", False)
        recent_asts = _checkpoint.get("recent_asts", [])
        start_iter = _checkpoint.get("iteration", 0) + 1
        baseline_measured_at = _checkpoint.get("baseline_measured_at", 0)
        history = _checkpoint.get("history", [])
        last_hypothesis = _checkpoint.get("last_hypothesis", "")
        print(f"Restored checkpoint: iter {start_iter-1}, best={best_score:.2f} us, "
              f"attempts={attempts}")
    else:
        initial_full_repeats = int(config.get("full_bench_repeats", 3))
        print(f"No checkpoint - measuring baseline (min of {initial_full_repeats} full benches)...")
        baseline_code = algo_path(algo).read_text()
        baseline_result = run_bench_repeated(algo, fast=False, repeats=initial_full_repeats)
        if baseline_result is None:
            print("ERROR: baseline bench failed")
            sys.exit(1)
        best_score = composite_score(baseline_result)
        attempts: dict = fresh_attempts()
        allow_widen = False
        recent_asts: list = []
        start_iter = 1
        baseline_measured_at = 0
        history: list = []
        last_hypothesis: str = ""
        baseline_path(algo).write_text(baseline_code)
        print(f"Baseline: {best_score:.2f} us")

    timeout_count = 0

    # iterations_override is a relative count from wherever we resume
    stop_at = (start_iter + iterations_override - 1) if iterations_override else None

    for i in range(start_iter, start_iter + 10_000):
        config = load_config()
        config = _apply_overrides(config, model_override, provider_override)
        max_iterations = int(config.get("max_iterations", 50))
        iteration_timeout = int(config.get("iteration_timeout", 360))
        checkpoint_interval = int(config.get("checkpoint_interval", 5))
        full_bench_repeats = int(config.get("full_bench_repeats", 3))

        if stop_at is not None and i > stop_at:
            print(f"\nReached requested iterations={iterations_override}. Done.")
            break
        if stop_at is None and i > max_iterations:
            print(f"\nReached max_iterations={max_iterations}. Done.")
            break

        # Re-measure baseline every 10 iters (drift control).
        # CRITICAL: best_score is MONOTONIC - drift refresh may only *tighten* it.
        # If the fresh measurement is worse, it's noise (CPU contention, GC, etc),
        # not a real regression in baseline_code. We keep the stored best so that
        # candidates are judged against the tightest valid measurement of the current
        # baseline code, preventing false wins from noise-inflated baselines.
        if i - baseline_measured_at >= 10:
            print(f"  [drift control] Re-measuring baseline (min of {full_bench_repeats})...")
            fresh = run_bench_repeated(algo, fast=False, repeats=full_bench_repeats)
            if fresh is not None:
                fresh_score = composite_score(fresh)
                if fresh_score < best_score:
                    baseline_result = fresh
                    best_score = fresh_score
                    baseline_measured_at = i
                    print(f"  Baseline tightened: {best_score:.2f} us")
                else:
                    baseline_measured_at = i
                    print(f"  Drift check: fresh={fresh_score:.2f} us worse than stored "
                          f"best={best_score:.2f} us - keeping stored (noise)")

        attempts_per_hypothesis = int(config.get("attempts_per_hypothesis", 3))

        # Pick hypothesis
        hypothesis = next_hypothesis(attempts, attempts_per_hypothesis, allow_widen)
        if hypothesis is None:
            exhausted_names = [h["name"] for h in HYPOTHESIS_TYPES
                               if attempts.get(h["name"], 0) >= attempts_per_hypothesis]
            if timeout_count > 0:
                print(f"\nAll hypotheses exhausted ({timeout_count} timeout(s) observed). "
                      f"Spent {attempts_per_hypothesis} attempts on each of: {exhausted_names}. "
                      f"{algo} locally optimal or stuck.")
            else:
                print(f"\nAll hypotheses exhausted. Spent {attempts_per_hypothesis} attempts on "
                      f"each of: {exhausted_names}. {algo} locally optimal.")
            break

        # Clear recent_asts on hypothesis switch (avoids cross-hypothesis stale contamination)
        if hypothesis["name"] != last_hypothesis:
            if recent_asts:
                print(f"  Hypothesis switched {last_hypothesis!r} → {hypothesis['name']!r} - clearing recent_asts")
                recent_asts = []
            last_hypothesis = hypothesis["name"]

        active_model = config.get("active_model", "")
        used_model = active_model  # updated after successful LLM call
        noise_tolerance = float(config.get("noise_tolerance", 1.05))

        bottleneck_metric = identify_bottleneck(baseline_result)
        bottleneck_fn_name = test_metric_to_fn_name(bottleneck_metric, baseline_code)
        bottleneck_us = baseline_result["metrics"][bottleneck_metric]
        total_us = sum(baseline_result["metrics"].values())

        print(f"--- Iter {i}/{max_iterations} | hypothesis={hypothesis['name']} | "
              f"bottleneck={bottleneck_fn_name} ({bottleneck_us:.0f} us) | "
              f"best={best_score:.2f} us ---")

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(iteration_timeout)

        try:
            # Build clean-room prompt
            prompt = build_prompt(
                baseline_code, bottleneck_fn_name, bottleneck_us, total_us,
                hypothesis, algo, env_manifest, program_text,
            )

            candidate, used_model, raw_response = get_llm_proposal(prompt, config)
            save_candidate(algo, i, hypothesis["name"], used_model, raw_response, candidate)

            h_name = hypothesis["name"]

            def _bump_attempt():
                attempts[h_name] = attempts.get(h_name, 0) + 1

            # Purity gate
            purity_err = check_purity(candidate)
            if purity_err:
                _bump_attempt()
                print(f"  Purity rejected: {purity_err} "
                      f"(attempts: {attempts[h_name]}/{attempts_per_hypothesis})")
                log_result(algo, {
                    "iteration": i, "hypothesis": h_name,
                    "bottleneck_fn": bottleneck_fn_name, "result": "purity_rejected",
                    "error": purity_err, "model": used_model, "best_score": best_score,
                    "attempts": attempts[h_name],
                    "timestamp": datetime.now().isoformat(),
                })
                continue

            # Correctness gate - candidate must produce the same output as
            # baseline on deterministic test inputs. Drift = attempt failure.
            from tools.autoresearch.correctness import check_correctness
            correctness_err = check_correctness(candidate, baseline_code, algo)
            if correctness_err:
                _bump_attempt()
                print(f"  Correctness rejected: {correctness_err} "
                      f"(attempts: {attempts[h_name]}/{attempts_per_hypothesis})")
                log_result(algo, {
                    "iteration": i, "hypothesis": h_name,
                    "bottleneck_fn": bottleneck_fn_name, "result": "correctness_rejected",
                    "error": correctness_err, "model": used_model, "best_score": best_score,
                    "attempts": attempts[h_name],
                    "timestamp": datetime.now().isoformat(),
                })
                continue

            # AST staleness gate - scope_widen compares whole file (helpers may change)
            use_whole_file = h_name == "scope_widen"
            if is_stale(candidate, baseline_code, recent_asts, bottleneck_fn_name, whole_file=use_whole_file):
                _bump_attempt()
                print(f"  Stale: structurally identical to baseline or recent attempt "
                      f"(attempts: {attempts[h_name]}/{attempts_per_hypothesis})")
                log_result(algo, {
                    "iteration": i, "hypothesis": h_name,
                    "bottleneck_fn": bottleneck_fn_name, "result": "stale",
                    "model": used_model, "best_score": best_score,
                    "attempts": attempts[h_name],
                    "timestamp": datetime.now().isoformat(),
                })
                continue

            if use_whole_file:
                recent_asts = (recent_asts + [file_ast_dump(candidate)])[-3:]
            else:
                recent_asts = (recent_asts + [fn_ast_dump(candidate, bottleneck_fn_name)])[-3:]

            # Isolated bench - baseline always restored in finally
            result = None
            tier = "failed"
            try:
                algo_path(algo).write_text(candidate)
                result, tier = run_bench_tiered(algo, best_score, noise_tolerance, full_bench_repeats)
            finally:
                algo_path(algo).write_text(baseline_code)

            score = composite_score(result) if result else float("inf")
            print(f"  Score: {score:.2f} us [{tier}] (best: {best_score:.2f} us)")

            entry = {
                "iteration": i,
                "hypothesis": h_name,
                "bottleneck_fn": bottleneck_fn_name,
                "composite_us": score,
                "best_score_before": best_score,
                "tier": tier,
                "metrics": result["metrics"] if result else {},
                "timestamp": datetime.now().isoformat(),
                "model": used_model,
                "attempts_before": attempts.get(h_name, 0),
            }

            if score < best_score and result is not None and tier != "failed":
                # WIN - reset all attempts, new baseline gets fresh budget
                entry["result"] = "win"
                print(f"  WIN: {score:.2f} us via {h_name} on {bottleneck_fn_name}")
                baseline_code = candidate
                baseline_result = result
                best_score = score
                baseline_measured_at = i
                baseline_path(algo).write_text(baseline_code)
                attempts = fresh_attempts()
                allow_widen = False
                recent_asts = []
                print(f"  Baseline promoted. Attempts reset.")
            else:
                # Non-win: 1 attempt consumed. Any outcome counts the same.
                _bump_attempt()
                if result is None or tier == "failed":
                    entry["result"] = "bench_failed"
                    reason = "bench failed (infra)"
                elif tier == "rejected":
                    entry["result"] = "slow_reject_quick"
                    reason = "slow (quick)"
                else:
                    entry["result"] = "slow_reject_full"
                    reason = "no improvement (full)"
                print(f"  {reason}. Attempts: {attempts[h_name]}/{attempts_per_hypothesis}")

            history.append(entry)
            log_result(algo, entry)

            # Unlock scope_widen when all standard hypotheses have been fully attempted
            all_standard_done = all(
                attempts.get(h["name"], 0) >= attempts_per_hypothesis
                for h in HYPOTHESIS_TYPES if h["name"] != "scope_widen"
            )
            if all_standard_done and not allow_widen:
                allow_widen = True
                attempts["scope_widen"] = 0
                print(f"  Unlocked scope_widen (all standard hypotheses exhausted)")

        except IterationTimeout:
            hn = hypothesis["name"]
            attempts[hn] = attempts.get(hn, 0) + 1
            print(f"  TIMEOUT - restoring baseline "
                  f"(attempts: {attempts[hn]}/{attempts_per_hypothesis})")
            algo_path(algo).write_text(baseline_code)
            timeout_count += 1
        except Exception as e:
            hn = hypothesis["name"]
            attempts[hn] = attempts.get(hn, 0) + 1
            print(f"  Error: {e} - restoring baseline "
                  f"(attempts: {attempts[hn]}/{attempts_per_hypothesis})")
            algo_path(algo).write_text(baseline_code)
        finally:
            signal.alarm(0)
            _checkpoint = {
                "iteration": i,
                "best_score": best_score,
                "baseline_code": baseline_code,
                "baseline_result": baseline_result,
                "baseline_measured_at": baseline_measured_at,
                "attempts": attempts,
                "allow_scope_widen": allow_widen,
                "recent_asts": recent_asts,
                "last_hypothesis": last_hypothesis,
                "history": history,
            }

        if i % checkpoint_interval == 0:
            save_checkpoint(algo)
            print(f"  Checkpoint saved.")

        time.sleep(int(config.get("sleep_between", 2)))

    print(f"\n=== Done ===")
    print(f"Best: {best_score:.2f} us  |  Iterations: {i - start_iter + 1}")
    save_checkpoint(algo)


def main():
    parser = argparse.ArgumentParser(description="Algo autoresearch - scientific ratchet")
    parser.add_argument("--algo", default=None, help="Algo name (overrides config)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Run this many iterations from where we resume (relative, not absolute)")
    parser.add_argument("--model", default=None,
                        help="Override active_model from config (e.g. 'glm-5.1:cloud'). "
                             "Auto-selects the provider that owns this model.")
    parser.add_argument("--provider", default=None,
                        help="Override active_provider from config (e.g. 'local_ollama')")
    args = parser.parse_args()
    run_loop(
        algo_override=args.algo,
        iterations_override=args.iterations,
        model_override=args.model,
        provider_override=args.provider,
    )


if __name__ == "__main__":
    main()
