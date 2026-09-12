"""Embedder protocol for text embedding models."""

import numpy as np


class Embedder:
    """Base interface for text embedding models.

    Subclasses must implement encode_documents (for storage) and
    encode_queries (for search). Asymmetric models (EmbeddingGemma)
    override both with different prefixes. Symmetric models (model2vec)
    use the same encoding for both.
    """

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def dims(self) -> int:
        """Output embedding dimensionality."""
        raise NotImplementedError

    def encode_documents(self, texts: list[str], titles: list[str | None] | None = None) -> np.ndarray:
        """Encode texts for storage. Returns shape (len(texts), dims), dtype float32."""
        raise NotImplementedError

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Encode texts for search/retrieval. Returns shape (len(texts), dims), dtype float32."""
        raise NotImplementedError
