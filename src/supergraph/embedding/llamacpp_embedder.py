"""llama.cpp embedder via llama-cpp-python - GGUF models, fine-grained quant."""

import numpy as np

from supergraph.embedding.base import Embedder
from supergraph.embedding.postprocess import l2_normalize, truncate_dims


class LlamaCppEmbedder(Embedder):
    """GGUF model embedder via llama-cpp-python.

    Handles encoder and decoder embedding models. GPU via n_gpu_layers=-1
    (native CUDA/Metal, no onnxruntime, no cu12 wheels).
    Supports Q2 through Q8 and FP16 GGUF quantization tiers.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        output_dims: int | None = None,
        query_prefix: str = "",
        doc_prefix_template: str = "",
    ):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "LlamaCppEmbedder requires llama-cpp-python. "
                "Install with: pip install 'supergraph[embedders-extra]' "
                "(or 'supergraph[vision]' which ships the same wheel)."
            ) from e

        self._model = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self._output_dims = output_dims or self._model.n_embd()
        self._base_dims = self._model.n_embd()
        self._query_prefix = query_prefix
        self._doc_prefix_template = doc_prefix_template

    @property
    def name(self) -> str:
        return "llama-cpp"

    @property
    def dims(self) -> int:
        return self._output_dims

    def encode_documents(
        self,
        texts: list[str],
        titles: list[str | None] | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self._output_dims), dtype=np.float32)
        prefixed = []
        for i, text in enumerate(texts):
            title = titles[i] if titles and i < len(titles) and titles[i] else "none"
            prefix = self._doc_prefix_template.format(title=title) if self._doc_prefix_template else ""
            prefixed.append(f"{prefix}{text}")
        return self._encode(prefixed)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._output_dims), dtype=np.float32)
        prefixed = [f"{self._query_prefix}{t}" for t in texts]
        return self._encode(prefixed)

    def _encode(self, texts: list[str]) -> np.ndarray:
        raw_embeds = [self._model.embed(t) for t in texts]
        flattened = [r[0] if isinstance(r, list) and len(r) > 0 and isinstance(r[0], list) else r for r in raw_embeds]
        vecs = np.array(flattened, dtype=np.float32)

        if self._output_dims < vecs.shape[1]:
            return truncate_dims(vecs, self._output_dims)
        return l2_normalize(vecs)
