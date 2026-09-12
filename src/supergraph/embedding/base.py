
import numpy as np


class Embedder:

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def dims(self) -> int:
        raise NotImplementedError

    def encode_documents(self, texts: list[str], titles: list[str | None] | None = None) -> np.ndarray:
        raise NotImplementedError

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError
