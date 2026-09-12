import subprocess, sys, os, time, shutil


RUN_TAG         = "supergraph-jina-v5-small"

EMBEDDER_REPO   = "jinaai/jina-embeddings-v5-text-small-retrieval"
EMBEDDER_POOLING    = "last_token"
EMBEDDER_MAX_LEN    = "2048"
EMBEDDER_DIMS       = "1024"

NER_REPO        = "onnx-community/TinyBERT-finetuned-NER-ONNX"

DATASET_REPO    = "xiaowu0162/longmemeval-cleaned"
DATASET_VARIANT = "s"

REPO_URL        = "https://github.com/orkait/supergraph.git"
REPO_BRANCH     = "main"

GPU_MEM_GB      = "12"
EMBED_BATCH     = "256"

EMBEDDER_KAGGLE_SLUG = "superkaiii/jina-v5-small-onnx"
NER_KAGGLE_SLUG      = "superkaiii/tinybert-ner-onnx"

WORKING         = "/kaggle/working"
SUPERGRAPH_DIR  = f"{WORKING}/supergraph"
EMBEDDER_DIR    = f"{WORKING}/embedder-model"
DATASET_DIR     = f"{WORKING}/dataset/longmemeval"
NER_DIR         = f"{SUPERGRAPH_DIR}/models/ner"
RESULTS_DIR     = f"{WORKING}/results"
CONFIG_PATH     = f"{SUPERGRAPH_DIR}/benchmarks/supergraph.json"
HF_TOKEN_FILE   = "/kaggle/input/hf-token-private/hf_token.txt"
KAGGLE_INPUT    = "/kaggle/input"


def log(msg):
    print(f"[{RUN_TAG}] {msg}")


def run_cmd(cmd, label, env=None):
    start = time.time()
    log(f"RUN {label}...")
    try:
        subprocess.check_call(cmd, env=env)
        log(f"OK  {label} ({time.time() - start:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        log(f"FAIL {label}: exit {e.returncode}")
        return False
    except Exception as e:
        log(f"FAIL {label}: {e}")
        return False


def download_with_retry(repo_id, local_dir, token, label, repo_type="model", max_retries=3):
    from huggingface_hub import snapshot_download
    for attempt in range(1, max_retries + 1):
        try:
            log(f"DL  {label} (attempt {attempt}/{max_retries})...")
            os.makedirs(local_dir, exist_ok=True)
            snapshot_download(repo_id, local_dir=local_dir, repo_type=repo_type, token=token)
            log(f"OK  {label}")
            return True
        except Exception as e:
            if attempt < max_retries:
                delay = 5 * attempt
                log(f"RETRY {label}: {e} - waiting {delay}s...")
                time.sleep(delay)
            else:
                log(f"FAIL {label}: {e} (exhausted {max_retries} retries)")
                return False
    return False


def cleanup():
    for d in [SUPERGRAPH_DIR, RESULTS_DIR]:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
                log(f"CLEANUP removed {d}")
            except Exception as e:
                log(f"WARN could not remove {d}: {e}")


def auth_hf_token():
    token = ""

    if os.path.exists(HF_TOKEN_FILE):
        try:
            token = open(HF_TOKEN_FILE).read().strip()
            log("OK  HF_TOKEN from private dataset")
        except Exception as e:
            log(f"WARN HF_TOKEN file unreadable: {e}")

    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            client = UserSecretsClient()
            for name in ["HF_TOKEN", "huggingface_token", "HF_HUB_TOKEN"]:
                try:
                    token = client.get_secret(name)
                    if token:
                        log(f"OK  HF_TOKEN from Kaggle Secrets ({name})")
                        break
                except Exception:
                    continue
        except Exception as e:
            log(f"WARN Kaggle Secrets unavailable: {e}")

    if token:
        os.environ["HF_TOKEN"] = token
        log(f"OK  HF_TOKEN active (prefix: {token[:4]}...)")
    else:
        log("WARN no HF_TOKEN - rate limits apply")

    return token


def install_deps():
    core = [
        "numpy>=1.24", "scipy>=1.10", "lark>=1.1", "usearch>=2.0",
        "model2vec>=0.4", "msgspec>=0.18", "croniter>=6.0", "orjson>=3.11.8",
        "psutil>=5.9", "tokenizers>=0.20", "onnxruntime-gpu>=1.23", "onnx>=1.14",
        "huggingface_hub",
    ]
    if not run_cmd([sys.executable, "-m", "pip", "install", "-q"] + core, "pip install core deps"):
        return False
    if not run_cmd([sys.executable, "-m", "pip", "install", "-q",
                    "--no-deps", "--force-reinstall", "onnxruntime-gpu>=1.23"],
                   "pip reinstall onnxruntime-gpu"):
        return False
    return True


def hf_login(token):
    if not token:
        log("SKIP HF login (no token)")
        return
    try:
        from huggingface_hub import login
        login(token=token)
        log("OK  HF Hub login")
    except Exception as e:
        log(f"WARN HF login failed: {e}")


def clone_repo():
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    return run_cmd(
        ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, SUPERGRAPH_DIR],
        f"git clone {REPO_BRANCH}", env=env
    )


def use_kaggle_dataset(slug, target_dir, label):
    if not slug:
        return False
    slug_name = slug.split("/")[-1]
    input_path = os.path.join(KAGGLE_INPUT, slug_name)
    if os.path.exists(input_path):
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        if not os.path.exists(target_dir):
            os.symlink(input_path, target_dir)
        log(f"OK  {label} from Kaggle dataset ({input_path})")
        return True
    return False


def download_assets(token):
    ok = True

    if not use_kaggle_dataset(EMBEDDER_KAGGLE_SLUG, EMBEDDER_DIR, "embedder"):
        ok &= download_with_retry(EMBEDDER_REPO, EMBEDDER_DIR, token,
                                  f"embedder ({EMBEDDER_REPO})")

    ok &= download_with_retry(DATASET_REPO, DATASET_DIR, token,
                              f"dataset ({DATASET_REPO})", repo_type="dataset")

    if not use_kaggle_dataset(NER_KAGGLE_SLUG, NER_DIR, "NER model"):
        ok &= download_with_retry(NER_REPO, NER_DIR, token,
                                  f"NER model ({NER_REPO})")

    return ok


def setup_env():
    if not os.path.exists(CONFIG_PATH):
        log(f"FAIL config not found: {CONFIG_PATH}")
        return False

    os.environ["SUPERGRAPH_CONFIG"] = CONFIG_PATH
    sys.path.insert(0, f"{SUPERGRAPH_DIR}/src")
    sys.path.insert(0, SUPERGRAPH_DIR)

    sys.argv = [
        "bench",
        "--system",               "supergraph",
        "--dataset",              "longmemeval",
        "--data-path",            DATASET_DIR,
        "--variant",              DATASET_VARIANT,
        "--embedder",             "onnx",
        "--embedder-model-dir",   EMBEDDER_DIR,
        "--embedder-pooling",     EMBEDDER_POOLING,
        "--embedder-max-length",  EMBEDDER_MAX_LEN,
        "--embedder-output-dims", EMBEDDER_DIMS,
        "--entity-model-dir",     NER_DIR,
        "--gpu",
        "--gpu-mem-limit-gb",     GPU_MEM_GB,
        "--embed-batch-size",     EMBED_BATCH,
        "--out-dir",              RESULTS_DIR,
        "--run-tag",              RUN_TAG,
    ]
    log("OK  env + sys.argv configured")
    return True


def run_benchmark():
    try:
        from benchmarks.framework.docker_runner import main
        log("RUN benchmark starting...")
        return main()
    except ImportError as e:
        log(f"FAIL import docker_runner: {e}")
        return 1
    except Exception as e:
        log(f"FAIL benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    print(f"\n{'='*60}")
    log(f"run={RUN_TAG}  branch={REPO_BRANCH}  embedder={EMBEDDER_REPO.split('/')[-1]}")
    print(f"{'='*60}\n")

    cleanup()
    token = auth_hf_token()

    if not install_deps():
        sys.exit(1)

    hf_login(token)

    if not clone_repo():
        sys.exit(1)

    if not download_assets(token):
        sys.exit(1)

    if not setup_env():
        sys.exit(1)

    sys.exit(run_benchmark())


if __name__ == "__main__":
    main()
