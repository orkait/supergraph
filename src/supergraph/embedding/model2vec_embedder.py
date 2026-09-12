"""Model2Vec embedder: lightweight, numpy-only, 50k texts/sec on CPU."""

from pathlib import Path
import numpy as np
from supergraph.embedding.base import Embedder

_model_cache: dict = {}


class Model2VecEmbedder(Embedder):
    """Default embedder. 30MB, numpy-only, zero-config.

    Symmetric model - queries and documents use the same encoding.
    Model instance is cached at module level so repeated construction is free.
    """

    def __init__(self, model_name: str = "minishlab/M2V_base_output", cache_dir: str | None = None):
        cache_key = (model_name, cache_dir)
        if cache_key not in _model_cache:
            from model2vec import StaticModel

            # If cache_dir is provided and model exists there, try loading from local
            if cache_dir:
                local_path = Path(cache_dir) / model_name.split("/")[-1]
                if local_path.exists():
                    _model_cache[cache_key] = StaticModel.from_pretrained(str(local_path))
                else:
                    # Pass cache_dir as a kwarg so two threads constructing
                    # Model2Vec instances with different cache_dirs don't
                    # race through the shared HF_HOME env var (bug #68).
                    # Pre-fix, the race window between set and restore
                    # let thread B read thread A's temporary HF_HOME. The
                    # underlying huggingface_hub.snapshot_download accepts
                    # a cache_dir kwarg that bypasses env entirely; pass
                    # it through where StaticModel supports it, otherwise
                    # fall back to the env-based path.
                    try:
                        _model_cache[cache_key] = StaticModel.from_pretrained(
                            model_name, cache_dir=str(cache_dir),
                        )
                    except TypeError:
                        # Older StaticModel without cache_dir kwarg —
                        # preserve prior behavior but shrink the race
                        # window to just the download call itself.
                        import os
                        old_hf_home = os.environ.get("HF_HOME")
                        os.environ["HF_HOME"] = str(cache_dir)
                        try:
                            _model_cache[cache_key] = StaticModel.from_pretrained(model_name)
                        finally:
                            if old_hf_home is not None:
                                os.environ["HF_HOME"] = old_hf_home
                            else:
                                os.environ.pop("HF_HOME", None)
            else:
                _model_cache[cache_key] = StaticModel.from_pretrained(model_name)
                
        self._model = _model_cache[cache_key]
        self._name = "model2vec"

    @property
    def name(self) -> str:
        return self._name

    @property
    def dims(self) -> int:
        return self._model.dim

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dims), dtype=np.float32)
        return self._model.encode(texts).astype(np.float32)

    def encode_documents(self, texts: list[str], titles: list[str | None] | None = None) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)
