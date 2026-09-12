
import time

from supergraph.core.runtime import RuntimeState
from supergraph.core.errors import SuperGraphError
from supergraph.core.types import Result

from supergraph.dsl.sys import (
    SYS_DISPATCH,
    SysQueryHandlers,
    SysSchemaHandlers,
    SysLifecycleHandlers,
    SysPipelineHandlers,
    SysCronHandlers,
    SysEvolveHandlers,
)


class SystemExecutor(
    SysQueryHandlers,
    SysSchemaHandlers,
    SysLifecycleHandlers,
    SysPipelineHandlers,
    SysCronHandlers,
    SysEvolveHandlers,
):

    def __init__(
        self,
        runtime: RuntimeState,
        retention: dict | None = None,
        cron=None,
        evolution_engine=None,
    ):
        self._runtime = runtime
        self._retention = retention or {}
        self._cron = cron
        self._evolution_engine = evolution_engine
        self._eviction_target_ratio = 0.8
        self._start_time = time.time()
        self._duplicate_threshold_override: float | None = None
        self._protected_kinds: set[str] | None = None
        self._wal_manager = None
        self._executor = None

    @property
    def store(self):
        return self._runtime.store

    @property
    def schema(self):
        return self._runtime.schema

    @property
    def conn(self):
        return self._runtime.conn

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
        result.elapsed_us = (time.perf_counter_ns() - start) // 1000
        return result

    def _dispatch(self, ast) -> Result:
        handler = SYS_DISPATCH.get(type(ast))
        if handler is None:
            raise SuperGraphError(f"Unknown system command: {type(ast).__name__}")
        return handler(self, ast)
