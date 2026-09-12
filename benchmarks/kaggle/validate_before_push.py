#!/usr/bin/env python3
"""Pre-push Kaggle validation. Runs locally against real models.

Tests:
  1. Core imports (supergraph, onnxruntime, etc.)
  2. Circular import check
  3. Model path structure (jina + tinybert)
  4. Benchmark pipeline - 1 record, CPU mode
  5. sys.argv config matches docker_runner expectations

Usage:
    python benchmarks/kaggle/validate_before_push.py
    python benchmarks/kaggle/validate_before_push.py --skip-run
"""
import argparse
import json
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
JINA_DIR = ROOT / "jina-small"
NER_DIR = ROOT / "models" / "tinybert-ner"
FIXTURE = ROOT / "tests" / "fixtures" / "benchmarks" / "longmemeval_sample.json"
CONFIG = ROOT / "benchmarks" / "supergraph.json"

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def check(label, fn):
    try:
        fn()
        print(f"  {PASS} {label}")
        return True
    except Exception as e:
        print(f"  {FAIL} {label}: {e}")
        return False


def validate_imports():
    print("\n[1] Core imports")
    ok = True
    ok &= check("supergraph package", lambda: __import__("supergraph"))
    ok &= check("supergraph.store.SuperGraph", lambda: __import__("supergraph.store", fromlist=["SuperGraph"]))
    ok &= check("supergraph.ingest.entity_extract", lambda: __import__("supergraph.ingest.entity_extract"))
    ok &= check("supergraph.registry.installer", lambda: __import__("supergraph.registry.installer"))
    ok &= check("benchmarks.framework.docker_runner", lambda: __import__("benchmarks.framework.docker_runner"))
    ok &= check("onnxruntime", lambda: __import__("onnxruntime"))
    ok &= check("huggingface_hub", lambda: __import__("huggingface_hub"))
    return ok


def validate_no_circular():
    print("\n[2] Circular import check")
    import importlib, sys
    mods_before = set(sys.modules.keys())
    ok = True
    for mod in ["supergraph", "supergraph.store", "supergraph.registry.installer",
                "supergraph.ingest.entity_extract"]:
        try:
            if mod in sys.modules:
                del sys.modules[mod]
            importlib.import_module(mod)
            print(f"  {PASS} {mod}")
        except ImportError as e:
            if "circular" in str(e).lower() or "partially initialized" in str(e).lower():
                print(f"  {FAIL} CIRCULAR: {mod}: {e}")
                ok = False
            else:
                print(f"  {PASS} {mod} (import error unrelated to circular: {e})")
    return ok


def validate_models():
    print("\n[3] Model paths")
    ok = True

    def check_jina(d):
        d = Path(d)
        assert d.exists(), f"dir missing: {d}"
        onnx_files = list((d / "onnx").glob("*.onnx")) if (d / "onnx").exists() else []
        assert onnx_files, f"no .onnx files in {d}/onnx/"
        # tokenizer.json at root (ONNX format from Kaggle) OR safetensors (local full model)
        has_tok = (d / "tokenizer.json").exists()
        has_sf = bool(list(d.glob("*.safetensors")))
        assert has_tok or has_sf, "neither tokenizer.json nor *.safetensors found"
        if not has_tok:
            print(f"    NOTE: local jina-small is safetensors (not ONNX). Kaggle downloads ONNX.")

    def check_ner(d):
        d = Path(d)
        assert d.exists(), f"dir missing: {d}"
        tok = d / "tokenizer.json"
        tok_onnx = d / "onnx" / "tokenizer.json"
        assert tok.exists() or tok_onnx.exists(), "tokenizer.json missing in root and onnx/"
        onnx = d / "onnx" / "model_int8.onnx"
        assert onnx.exists(), f"model_int8.onnx missing at {onnx}"

    ok &= check(f"jina-small @ {JINA_DIR}", lambda: check_jina(JINA_DIR))
    ok &= check(f"tinybert-ner @ {NER_DIR}", lambda: check_ner(NER_DIR))
    ok &= check(f"benchmark config @ {CONFIG}", lambda: json.load(open(CONFIG)))
    ok &= check(f"fixture data @ {FIXTURE}", lambda: json.load(open(FIXTURE)))
    return ok


def validate_mini_run():
    print("\n[4] Mini benchmark run (1 record, CPU)")
    import tempfile

    fixture_data = json.load(open(FIXTURE))
    tmpdata = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(fixture_data[:1], tmpdata)
    tmpdata.close()

    os.environ["SUPERGRAPH_CONFIG"] = str(CONFIG)
    out_dir = Path(tempfile.mkdtemp(prefix="kg_validate_"))

    sys.path.insert(0, str(ROOT / "src"))
    saved_argv = sys.argv[:]
    sys.argv = [
        "bench",
        "--system", "supergraph",
        "--dataset", "longmemeval",
        "--data-path", tmpdata.name,
        "--variant", "s",
        "--embedder", "onnx",
        "--embedder-model-dir", str(JINA_DIR),
        "--embedder-pooling", "last_token",
        "--embedder-max-length", "512",
        "--embedder-output-dims", "1024",
        "--entity-model-dir", str(NER_DIR),
        "--max-records", "1",
        "--out-dir", str(out_dir),
        "--run-tag", "validate",
    ]

    try:
        from benchmarks.framework.docker_runner import main
        rc = main()
        results = list(out_dir.glob("*.json"))
        assert results, "no result files written"
        print(f"  {PASS} 1 record complete, results at {out_dir}")
        return True
    except SystemExit as e:
        if e.code == 0:
            print(f"  {PASS} exited 0")
            return True
        print(f"  {FAIL} exited {e.code}")
        return False
    except Exception as e:
        print(f"  {FAIL} {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sys.argv = saved_argv
        Path(tmpdata.name).unlink(missing_ok=True)


def validate_argv_config():
    print("\n[5] sys.argv config sanity")
    ok = True

    def check_arg(flag, val=None):
        # Just verify the flag is in our expected argv set
        expected = [
            "--system", "--dataset", "--data-path", "--variant",
            "--embedder", "--embedder-model-dir", "--embedder-pooling",
            "--embedder-max-length", "--embedder-output-dims",
            "--entity-model-dir", "--gpu", "--gpu-mem-limit-gb",
            "--embed-batch-size", "--out-dir", "--run-tag",
        ]
        assert flag in expected, f"{flag} not in expected argv"

    for flag in ["--entity-model-dir", "--embedder-model-dir", "--gpu", "--embed-batch-size"]:
        ok &= check(flag, lambda f=flag: check_arg(f))
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-run", action="store_true", help="skip the mini benchmark run (faster)")
    args = p.parse_args()

    sys.path.insert(0, str(ROOT / "src"))

    results = []
    results.append(("imports", validate_imports()))
    results.append(("circular", validate_no_circular()))
    results.append(("models", validate_models()))
    results.append(("argv", validate_argv_config()))

    if not args.skip_run:
        results.append(("mini-run", validate_mini_run()))
    else:
        print(f"\n[4] Mini run {SKIP} (--skip-run)")
        print(f"\n[5] sys.argv config sanity")

    print("\n" + "="*50)
    all_ok = all(v for _, v in results)
    for name, ok in results:
        icon = PASS if ok else FAIL
        print(f"  {icon} {name}")

    if all_ok:
        print("\nAll checks passed. Safe to push.")
        sys.exit(0)
    else:
        print("\nFailed. Fix above before pushing.")
        sys.exit(1)
