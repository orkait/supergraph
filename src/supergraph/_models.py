"""Resolve optional model weights through the standard HuggingFace cache.

The HF cache location follows ``$HF_HOME`` / ``$HUGGINGFACE_HUB_CACHE``; we
do not introduce a parallel cache directory.
"""
from __future__ import annotations
import os
from pathlib import Path

_BONSAI_REPO = "superkaiii/Ternary-Bonsai-4B-GGUF"
_REPO_MARKER = "Ternary-Bonsai-4B-GGUF"


def _default_quant() -> str:
    return os.environ.get("SUPERGRAPH_BONSAI_QUANT", "TQ1_0").upper()


def _scan_cache_for_gguf(quant: str) -> Path | None:
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return None
    file_marker = f"-{quant}.gguf"
    for repo in scan_cache_dir().repos:
        if _REPO_MARKER not in str(repo.repo_id):
            continue
        for rev in repo.revisions:
            for f in rev.files:
                if f.file_name.endswith(file_marker):
                    p = Path(f.file_path)
                    if p.exists():
                        return p
    return None


def resolve_bonsai_gguf(
    quant: str | None = None,
    *,
    auto_download: bool = True,
) -> Path:
    """Return the local path to the Bonsai GGUF for ``quant``.

    Resolution order:
      1. Scan the HuggingFace cache for a matching file (fast, offline).
      2. If absent and ``auto_download`` is true, fetch from HuggingFace.
      3. Otherwise raise ``RuntimeError`` with actionable guidance.

    ``quant`` defaults to ``$SUPERGRAPH_BONSAI_QUANT`` or ``TQ1_0``. The HF
    cache location follows ``$HF_HOME`` / ``$HUGGINGFACE_HUB_CACHE`` per the
    library's own convention - we do not introduce a parallel cache.
    """
    q = (quant or _default_quant()).upper()

    cached = _scan_cache_for_gguf(q)
    if cached is not None:
        return cached

    if not auto_download:
        raise RuntimeError(
            f"Bonsai GGUF for quant={q!r} not found in HF cache. "
            "Run `supergraph pro setup` to download required models, "
            "or pass auto_download=True."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface-hub not installed; pip install 'supergraph[pro]'"
        ) from e

    fname = f"Ternary-Bonsai-4B-{q}.gguf"
    return Path(hf_hub_download(repo_id=_BONSAI_REPO, filename=fname))
