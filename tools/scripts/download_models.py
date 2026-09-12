"""Download models into ./models.

Embedders use the registry installer.  NER, reranker and model2vec default
are downloaded manually because they are not registered as embedders.

Usage:
    python -m tools.scripts.download_models
    # or from project root:
    uv run python3 tools/scripts/download_models.py
"""

from pathlib import Path
from huggingface_hub import hf_hub_download

from supergraph.registry.installer import install_embedder, set_cache_dir, is_installed

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def download_all() -> None:
    set_cache_dir(MODELS_DIR)

    # 1. Embedders (via registry installer)
    print("--- Installing embedders via registry ---")
    for model in [
        "jina-v5-nano-retrieval",
        "jina-v5-small-retrieval",
        "embeddinggemma-300m",
        "harrier-oss-v1-0.6b",
    ]:
        if not is_installed(model):
            print(f"  Installing {model}...")
            install_embedder(model)
        else:
            print(f"  {model} already installed.")

    # 2. NER: TinyBERT for entity extraction
    print("\n--- TinyBERT NER ---")
    ner_dir = MODELS_DIR / "tinybert-ner"
    ner_dir.mkdir(exist_ok=True)
    ner_repo = "onnx-community/TinyBERT-finetuned-NER-ONNX"
    ner_files = ["tokenizer.json", "config.json", "onnx/model_int8.onnx"]
    if not all((ner_dir / f).exists() for f in ner_files):
        for f in ner_files:
            print(f"  Downloading {f}...")
            hf_hub_download(ner_repo, f, local_dir=str(ner_dir))
    else:
        print("  Already present.")

    # 3. Reranker: Jina v3 GGUF
    print("\n--- Jina Reranker v3 GGUF ---")
    reranker_dir = MODELS_DIR / "jina-reranker-v3"
    reranker_dir.mkdir(exist_ok=True)
    reranker_repo = "jinaai/jina-reranker-v3-GGUF"
    reranker_files = ["jina-reranker-v3-Q8_0.gguf", "projector.safetensors"]
    if not all((reranker_dir / f).exists() for f in reranker_files):
        for f in reranker_files:
            print(f"  Downloading {f}...")
            hf_hub_download(reranker_repo, f, local_dir=str(reranker_dir))
    else:
        print("  Already present.")

    # 4. model2vec default embedder
    print("\n--- model2vec default (M2V_base_output) ---")
    m2v_dir = MODELS_DIR / "m2v-base"
    m2v_dir.mkdir(exist_ok=True)
    m2v_repo = "minishlab/M2V_base_output"
    m2v_files = ["model.safetensors", "config.json", "tokenizer.json"]
    if not all((m2v_dir / f).exists() for f in m2v_files):
        for f in m2v_files:
            try:
                print(f"  Downloading {f}...")
                hf_hub_download(m2v_repo, f, local_dir=str(m2v_dir))
            except Exception as e:
                print(f"  Skipped {f}: {e}")
    else:
        print("  Already present.")

    print("\nAll models ready in ./models")
    print(f"  Embedders: jina-v5-nano, jina-v5-small, embeddinggemma, harrier")
    print(f"  NER:       tinybert-ner (onnx-community/TinyBERT-finetuned-NER-ONNX)")
    print(f"  Reranker:  jina-reranker-v3-GGUF (jinaai/jina-reranker-v3-GGUF)")
    print(f"  Default:   M2V_base_output (minishlab/M2V_base_output)")


if __name__ == "__main__":
    download_all()
