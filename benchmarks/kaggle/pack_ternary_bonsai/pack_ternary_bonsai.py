from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


HF_REPO = "superkaiii/Ternary-Bonsai-4B-TQ1_0-GGUF"
SOURCE_REPO = "prism-ml/Ternary-Bonsai-4B-unpacked"
TARGET_QUANT = "TQ1_0"
OUTPUT_NAME = "Ternary-Bonsai-4B-TQ1_0.gguf"

SRC_DIR = Path("/kaggle/tmp/bonsai-src")
LC_DIR = Path("/kaggle/tmp/llama.cpp")
F16_GGUF = Path("/kaggle/tmp/bonsai-4b-f16.gguf")
WORKING = Path("/kaggle/working")
OUT_GGUF = WORKING / OUTPUT_NAME
REPORT = WORKING / "pack_report.json"


def run(cmd: list[str] | str, **kw) -> None:
    shell = isinstance(cmd, str)
    print(f"$ {cmd if shell else ' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, shell=shell, **kw)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_hf_token() -> str:
    primary = Path("/kaggle/input/hf-token-private/hf_token.txt")
    if primary.exists():
        return primary.read_text().strip()
    for name in ("hf_token.txt", "HF_TOKEN", "token"):
        hits = list(Path("/kaggle/input").rglob(name))
        if hits:
            return hits[0].read_text().strip()
    raise RuntimeError("HF token not found under /kaggle/input/")


def main() -> None:
    WORKING.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)

    hf_token = load_hf_token()
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    print("[1/6] pip deps")
    run("pip install -q 'huggingface_hub>=0.24' 'transformers>=4.44' 'safetensors>=0.4' 'torch>=2.3' 'sentencepiece' 'protobuf'")

    print("[2/6] download source F16 safetensors (Python API, not CLI)")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=SOURCE_REPO,
        local_dir=str(SRC_DIR),
        token=hf_token,
        allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"],
    )

    print("[3/6] clone + build llama.cpp llama-quantize target")
    if not LC_DIR.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", str(LC_DIR)])
    run(f"cd {LC_DIR} && cmake -B build -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=OFF && cmake --build build --target llama-quantize -j")

    print("[4/6] convert safetensors -> F16 GGUF")
    run([
        "python", str(LC_DIR / "convert_hf_to_gguf.py"),
        str(SRC_DIR),
        "--outfile", str(F16_GGUF),
        "--outtype", "f16",
    ])

    print("[5/6] quantize F16 -> TQ1_0 (the pack step)")
    run([
        str(LC_DIR / "build" / "bin" / "llama-quantize"),
        str(F16_GGUF),
        str(OUT_GGUF),
        TARGET_QUANT,
    ])

    if F16_GGUF.exists():
        F16_GGUF.unlink()
    shutil.rmtree(SRC_DIR, ignore_errors=True)

    print("[6/6] upload to HF Hub")
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(HF_REPO, repo_type="model", exist_ok=True, private=False)

    readme = WORKING / "README.md"
    readme.write_text(_readme())

    api.upload_file(
        path_or_fileobj=str(OUT_GGUF),
        path_in_repo=OUTPUT_NAME,
        repo_id=HF_REPO,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(readme),
        path_in_repo="README.md",
        repo_id=HF_REPO,
        repo_type="model",
    )

    report = {
        "source_repo": SOURCE_REPO,
        "target_repo": HF_REPO,
        "quant": TARGET_QUANT,
        "output_bytes": OUT_GGUF.stat().st_size,
        "output_sha256": sha256(OUT_GGUF),
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"\nDONE\n  local:  {OUT_GGUF} ({OUT_GGUF.stat().st_size / 1e9:.2f} GB)")
    print(f"  hub:    https://huggingface.co/{HF_REPO}")
    print(f"  report: {REPORT}")


def _readme() -> str:
    return f"""---
library_name: gguf
base_model: {SOURCE_REPO}
quantization_config:
  method: {TARGET_QUANT}
tags:
  - gguf
  - ternary
  - bonsai
  - llama.cpp
---

# Ternary-Bonsai-4B TQ1_0

Repack of [{SOURCE_REPO}](https://huggingface.co/{SOURCE_REPO}) to llama.cpp's native ternary format ({TARGET_QUANT}).

Ternary-Bonsai weights are trained as {{-1, 0, +1}}. {TARGET_QUANT} packs these losslessly at 1.6875 bits per weight with a single FP16 scale per group of 256 weights. Unlike Q2_K (which k-means clusters generic floats), {TARGET_QUANT} preserves the original trained weight values exactly.

## Quickstart

```python
from llama_cpp import Llama
m = Llama(model_path="{OUTPUT_NAME}", n_ctx=4096)
```

## Conversion pipeline

```
prism-ml/Ternary-Bonsai-4B-unpacked (F16 safetensors)
 -> convert_hf_to_gguf.py --outtype f16
 -> llama-quantize ... {TARGET_QUANT}
```

Built on Kaggle with ggerganov/llama.cpp master.
"""


if __name__ == "__main__":
    main()
