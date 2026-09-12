
import numpy as np

from supergraph.embedding.base import Embedder


class FastEmbedEmbedder(Embedder):

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads: int | None = None,
        providers: list[str] | None = None,
    ):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError(
                "FastEmbedEmbedder requires fastembed. "
                "Install with: pip install 'supergraphdb[embedders-extra]'"
            ) from e

        kwargs: dict = dict(model_name=model_name)
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if threads:
            kwargs["threads"] = threads
        if providers:
            kwargs["providers"] = providers

        self._model_name = model_name
        self._model = TextEmbedding(**kwargs)
        self._dims = int(self._model.embedding_size)

    @property
    def name(self) -> str:
        return f"fastembed:{self._model_name}"

    @property
    def dims(self) -> int:
        return self._dims

    def encode_documents(
        self,
        texts: list[str],
        titles: list[str | None] | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dims), dtype=np.float32)
        return np.array(list(self._model.passage_embed(texts)), dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dims), dtype=np.float32)
        return np.array(list(self._model.query_embed(texts)), dtype=np.float32)
