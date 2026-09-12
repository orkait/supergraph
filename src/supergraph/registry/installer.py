
import os
import subprocess
import sys
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
from supergraph.registry.models import get_model_info, SUPPORTED_MODELS


DEFAULT_CACHE_DIR = Path.home() / "supergraph-models"

_cache_dir_override: Path | None = None


def set_cache_dir(path: str | Path | None) -> None:
    global _cache_dir_override
    _cache_dir_override = Path(path) if path else None


def _effective_cache_dir() -> Path:
    if _cache_dir_override is not None:
        return _cache_dir_override
    env = os.environ.get("SUPERGRAPH_MODEL_CACHE")
    if env:
        return Path(env)
    return DEFAULT_CACHE_DIR


def get_model_dir(name: str) -> Path:
    return _effective_cache_dir() / name


def is_installed(name: str) -> bool:
    model_dir = get_model_dir(name)
    if not model_dir.exists():
        return False
    manifest = model_dir / "manifest.json"
    if not manifest.exists():
        return False
    has_gguf = any(model_dir.rglob("*.gguf"))
    has_onnx = any(model_dir.rglob("*.onnx"))
    if has_gguf:
        return True
    if has_onnx:
        return (model_dir / "tokenizer.json").exists()
    return False


def install_embedder(name: str, variant: str | None = None) -> Path:
    info = get_model_info(name)
    if info is None:
        available = list(SUPPORTED_MODELS.keys())
        raise ValueError(f"Unknown model: {name!r}. Available: {available}")

    variant = variant or info["default_variant"]
    model_dir = get_model_dir(name)
    model_dir.mkdir(parents=True, exist_ok=True)

    import os as _os
    allow_install = _os.environ.get("SUPERGRAPH_AUTO_PIP_INSTALL") == "1"
    for dep in info["deps"]:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            if dep == "onnxruntime":
                dep_name = _detect_onnx_package()
            else:
                dep_name = dep
            if not allow_install:
                raise RuntimeError(
                    f"Model {name!r} requires the {dep_name!r} Python package which is "
                    f"not installed. Install it first:\n"
                    f"    pip install {dep_name}\n"
                    f"or set SUPERGRAPH_AUTO_PIP_INSTALL=1 to let supergraph "
                    f"install missing deps automatically (not recommended outside "
                    f"development environments)."
                )
            print(f"Installing {dep_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep_name, "-q"])

    from huggingface_hub import hf_hub_download

    repo_id = info["repo_id"]
    variant_info = info["variants"][variant]
    family = info.get("family", "hf_onnx")

    if family == "hf_onnx":
        for tok_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
            try:
                hf_hub_download(repo_id, tok_file, local_dir=str(model_dir))
            except Exception as e:
                logger.debug("tokenizer file %r download skipped: %s", tok_file, e, exc_info=True)

    for f in variant_info["files"]:
        print(f"Downloading {f}...")
        hf_hub_download(repo_id, f, local_dir=str(model_dir))

    import json
    primary_file = variant_info["files"][0]
    manifest = {
        "name": name,
        "family": family,
        "variant": variant,
        "dims": info["base_dims"],
        "default_dims": info["default_dims"],
        "max_length": info["max_length"],
        "query_prefix": info["query_prefix"],
        "doc_prefix_template": info["doc_prefix_template"],
        "pooling": info.get("pooling", "mean"),
    }
    if family == "gguf":
        manifest["gguf_file"] = primary_file
    else:
        manifest["onnx_file"] = primary_file
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Installed {name} ({variant}) to {model_dir}")
    return model_dir


def uninstall_embedder(name: str) -> None:
    model_dir = get_model_dir(name)
    if model_dir.exists():
        shutil.rmtree(model_dir)
        print(f"Uninstalled {name}")
    else:
        print(f"{name} is not installed")


def load_installed_embedder(
    name: str,
    dims: int | None = None,
    providers: list[str] | str | None = None,
    n_gpu_layers: int = 0,
    gpu_mem_limit: int | None = None,
    max_length: int | None = None,
):
    model_dir = get_model_dir(name)
    if not is_installed(name):
        raise FileNotFoundError(
            f"Model {name!r} not installed. Run: supergraph install-embedder {name}"
        )

    import json
    manifest = json.loads((model_dir / "manifest.json").read_text())
    family = manifest.get("family", "hf_onnx")

    if family == "gguf":
        from supergraph.embedding.llamacpp_embedder import LlamaCppEmbedder
        gguf_file = manifest.get("gguf_file", "")
        model_path = str(model_dir / gguf_file)
        return LlamaCppEmbedder(
            model_path=model_path,
            n_ctx=max_length or manifest.get("max_length", 2048),
            n_gpu_layers=n_gpu_layers,
            output_dims=dims or manifest["default_dims"],
            query_prefix=manifest.get("query_prefix", ""),
            doc_prefix_template=manifest.get("doc_prefix_template", ""),
        )

    from supergraph.embedding.onnx_hf_embedder import OnnxHFEmbedder
    return OnnxHFEmbedder(
        model_dir=model_dir,
        output_dims=dims or manifest["default_dims"],
        query_prefix=manifest.get("query_prefix", ""),
        doc_prefix_template=manifest.get("doc_prefix_template", ""),
        max_length=max_length or manifest.get("max_length", 512),
        pooling_mode=manifest.get("pooling", "mean"),
        onnx_file=manifest.get("onnx_file"),
        providers=providers,
        gpu_mem_limit=gpu_mem_limit,
    )


def _detect_onnx_package() -> str:
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True)
        if result.returncode == 0:
            return "onnxruntime-gpu"
    except FileNotFoundError:
        pass
    return "onnxruntime"
