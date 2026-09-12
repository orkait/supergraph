
import numpy as np


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def truncate_dims(vectors: np.ndarray, target_dims: int) -> np.ndarray:
    truncated = vectors[:, :target_dims]
    return l2_normalize(truncated)
