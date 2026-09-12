"""VectorStore: HNSW vector index for semantic similarity search."""

from pathlib import Path
import numpy as np
from usearch.index import Index


class VectorStore:
    """HNSW vector index backed by usearch.
    
    Supports:
    - Binary Quantization (1-bit) for 32x RAM reduction.
    - Memory-Mapped (mmap) views for zero-RAM search on large datasets.
    """

    def __init__(self, dims: int, capacity: int = 1024, quantize_binary: bool = False, path: str | None = None):
        self._dims = dims
        self._quantize_binary = quantize_binary
        self._path = path
        self._readonly = False

        if quantize_binary:
            self._index = Index(ndim=dims, metric="hamming", dtype="b1")
        else:
            self._index = Index(ndim=dims, metric="cos", dtype="f32")

        if path and Path(path).exists():
            # Memory-mapped view for zero-RAM search. On first write,
            # we switch to an in-memory copy.
            self._index.view(str(path))
            self._readonly = True

        self._has_vector = np.zeros(capacity, dtype=bool)
        self._capacity = capacity

    def _ensure_writable(self) -> None:
        """Switch from mmap view to in-memory copy if a write is about to happen."""
        if not self._readonly or not self._path:
            return
        if not Path(self._path).exists():
            self._readonly = False
            return
        # Load from file into memory, preserving current state
        if self._quantize_binary:
            new_index = Index(ndim=self._dims, metric="hamming", dtype="b1")
        else:
            new_index = Index(ndim=self._dims, metric="cos", dtype="f32")
        new_index.load(self._path)
        self._index = new_index
        self._readonly = False

    @property
    def dims(self) -> int:
        return self._dims

    def add(self, slot: int, vector: np.ndarray) -> None:
        """Add or replace vector for a slot."""
        self._ensure_writable()
        if slot >= self._capacity:
            self.grow(max(slot + 1, self._capacity * 2))
            
        if self._quantize_binary:
            if vector.dtype == np.uint8:
                vec = vector.ravel()
            else:
                vec = np.asarray(vector, dtype=np.float32).ravel()
                if len(vec) != self._dims:
                    raise ValueError(f"Expected {self._dims} dims, got {len(vec)}")
                vec = np.packbits(vec > 0)
        else:
            vec = np.asarray(vector, dtype=np.float32).ravel()
            if len(vec) != self._dims:
                raise ValueError(f"Expected {self._dims} dims, got {len(vec)}")
                
        if self._has_vector[slot]:
            self._index.remove(slot)
        self._index.add(slot, vec)
        self._has_vector[slot] = True

    def remove(self, slot: int) -> None:
        """Remove vector for a slot."""
        self._ensure_writable()
        if slot < self._capacity and self._has_vector[slot]:
            self._index.remove(slot)
            self._has_vector[slot] = False

    def search(self, query: np.ndarray, k: int, mask: np.ndarray | None = None, oversample_factor: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors. Returns (slot_indices, distances).

        If mask provided, only slots where mask[slot]==True are considered.
        Uses oversampling + post-filter since usearch doesn't natively support masks.
        """
        count = self.count()
        if count == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        if self._quantize_binary:
            if query.dtype == np.uint8:
                search_query = query.ravel()
            else:
                query_float = np.asarray(query, dtype=np.float32).ravel()
                search_query = np.packbits(query_float > 0)
            def _normalize(d):
                return float(d) / (self._dims / 2.0)
        else:
            search_query = np.asarray(query, dtype=np.float32).ravel()
            def _normalize(d):
                return float(d)

        if mask is not None:
            # Adaptive oversample and filter
            current_oversample = min(k * oversample_factor, count)
            max_oversample = min(count, max(current_oversample * 16, 10000))
            
            while True:
                results = self._index.search(search_query, current_oversample)
                valid = []
                for key, dist in zip(results.keys, results.distances):
                    key = int(key)
                    if key < len(mask) and mask[key]:
                        valid.append((key, _normalize(dist)))
                        if len(valid) >= k:
                            break
                
                if len(valid) >= k or current_oversample >= max_oversample or current_oversample >= count:
                    break
                current_oversample = min(current_oversample * 2, count)

            if not valid:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            slots = np.array([v[0] for v in valid], dtype=np.int64)
            dists = np.array([v[1] for v in valid], dtype=np.float32)
            return slots, dists
        else:
            actual_k = min(k, count)
            results = self._index.search(search_query, actual_k)
            slots = np.array(results.keys, dtype=np.int64)
            dists = np.array([_normalize(d) for d in results.distances], dtype=np.float32)
            return slots, dists

    def has_vector(self, slot: int) -> bool:
        return slot < self._capacity and bool(self._has_vector[slot])

    def get_vector(self, slot: int) -> np.ndarray | None:
        """Get stored vector for a slot."""
        if not self.has_vector(slot):
            return None
        if self._quantize_binary:
            return np.array(self._index[slot], dtype=np.uint8)
        return np.array(self._index[slot], dtype=np.float32)

    def grow(self, new_capacity: int) -> None:
        old = self._has_vector
        self._has_vector = np.zeros(new_capacity, dtype=bool)
        self._has_vector[:len(old)] = old
        self._capacity = new_capacity

    def count(self) -> int:
        return int(np.sum(self._has_vector))

    @property
    def memory_bytes(self) -> int:
        """Approximate memory: vector data + HNSW graph overhead."""
        n = self.count()
        if self._quantize_binary:
            vec_bytes = (self._dims + 7) // 8
            return n * (vec_bytes + 64) + self._has_vector.nbytes
        else:
            return n * (self._dims * 4 + 64) + self._has_vector.nbytes

    def save(self, path: str | None = None) -> bytes | None:
        """Serialize index. Explicit `path` writes to file and returns None;
        no arg returns the serialised bytes regardless of ``self._path``.

        Callers that want a bytes buffer (e.g. SYS SNAPSHOT) must omit
        ``path`` - we used to fall back to ``self._path`` here, which meant
        snapshots on persistent stores got ``None`` back and then
        ``Index.load(None)`` crashed.
        """
        if path is not None:
            self._ensure_writable()
            self._index.save(str(path))
            return None
        return bytes(self._index.save(None))

    def load(self, data: bytes | str | Path) -> None:
        """Deserialize index from bytes or file path."""
        if isinstance(data, (str, Path)):
            self._path = str(data)
            # Memory-mapped view for zero-RAM search. Writable on demand via _ensure_writable().
            if self._quantize_binary:
                self._index = Index(ndim=self._dims, metric="hamming", dtype="b1")
            else:
                self._index = Index(ndim=self._dims, metric="cos", dtype="f32")
            self._index.view(str(data))
            self._readonly = True
        else:
            if self._quantize_binary:
                new_index = Index(ndim=self._dims, metric="hamming", dtype="b1")
            else:
                new_index = Index(ndim=self._dims, metric="cos", dtype="f32")
            new_index.load(data)
            self._index = new_index

        keys = list(self._index.keys)
        if keys:
            max_key = int(max(keys))
            new_capacity = max(self._capacity, max_key + 1)
            self._has_vector = np.zeros(new_capacity, dtype=bool)
            self._capacity = new_capacity
            for k in keys:
                self._has_vector[int(k)] = True
        else:
            self._has_vector = np.zeros(self._capacity, dtype=bool)
