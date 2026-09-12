
import time

from supergraph.core.runtime import RuntimeState
from supergraph.core.types import Result
from supergraph.dsl.visibility import VisibilityMixin
from supergraph.dsl.filtering import FilteringMixin


class ExecutorBase(VisibilityMixin, FilteringMixin):
    def __init__(self, runtime: RuntimeState,
                 ingest_root: str | None = None,
                 ingestor_registry=None, chunker=None):
        self._runtime = runtime
        self._ingest_root = ingest_root
        self._ingestor_registry = ingestor_registry
        self._chunker = chunker
        self.cost_threshold = 100_000
        self._embedder_dirty = False
        self._fts_full_text = True
        self._vision_model = "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf"
        self._vision_base_url: str | None = None
        self._vision_max_tokens = 512
        self._defer_embeddings: bool = False
        self._pending_embeddings: list[tuple[int, str]] = []
        self._embed_batch_size: int = 64
        self._similarity_threshold: float | None = None
        self._batch_vector_record: list[int] | None = None

    @property
    def store(self):
        return self._runtime.store

    @property
    def schema(self):
        return self._runtime.schema

    @property
    def _vector_store(self):
        return self._runtime.vector_store

    @property
    def _document_store(self):
        return self._runtime.document_store

    @property
    def _embedder(self):
        return self._runtime.embedder

    def execute(self, ast) -> Result:
        start = time.perf_counter_ns()
        result = self._dispatch(ast)
        elapsed = (time.perf_counter_ns() - start) // 1000
        result.elapsed_us = elapsed
        return result
