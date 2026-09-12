
import logging

from supergraph.core.runtime import RuntimeState

logger = logging.getLogger(__name__)


class OptimizerScheduler:

    def __init__(self, runtime: RuntimeState,
                 auto_optimize: bool = False, optimize_interval: int = 500,
                 compact_threshold: float = 0.2, string_gc_threshold: float = 3.0,
                 cache_gc_threshold: int = 200, evolution_engine=None):
        self._runtime = runtime
        self._auto_optimize = auto_optimize
        self._optimize_interval = optimize_interval
        self._compact_threshold = compact_threshold
        self._string_gc_threshold = string_gc_threshold
        self._cache_gc_threshold = cache_gc_threshold
        self._optimizing = False
        self._needs_optimize = False
        self._write_counter = 0
        self._evolution_engine = evolution_engine

    @property
    def _store(self):
        return self._runtime.store

    @property
    def _vector_store(self):
        return self._runtime.vector_store

    @property
    def _document_store(self):
        return self._runtime.document_store

    @property
    def _schema(self):
        return self._runtime.schema

    @property
    def _conn(self):
        return self._runtime.conn

    @property
    def optimizing(self) -> bool:
        return self._optimizing

    def on_write(self) -> None:
        self._write_counter += 1
        if self._auto_optimize and self._write_counter % self._optimize_interval == 0:
            self._check_health()

    def maybe_optimize(self) -> None:
        if not self._needs_optimize:
            return
        self._optimizing = True
        try:
            from supergraph.core.optimizer import optimize_all
            optimize_all(
                self._store, self._vector_store, self._document_store,
                schema=self._schema, conn=self._conn,
            )
        except Exception as e:
            logger.debug("auto-optimize failed: %s", e)
        finally:
            self._optimizing = False
            self._needs_optimize = False

    def _check_health(self) -> None:
        try:
            from supergraph.core.optimizer import health_check, needs_optimization
            health = health_check(self._store, self._vector_store, self._document_store)
            if needs_optimization(health,
                                  compact_threshold=self._compact_threshold,
                                  string_gc_threshold=self._string_gc_threshold,
                                  cache_gc_threshold=self._cache_gc_threshold):
                self._needs_optimize = True
            from supergraph.core.memory import check_ceiling_accurate
            if check_ceiling_accurate(self._store, self._vector_store, self._store._ceiling_bytes):
                from supergraph.core.optimizer import evict_oldest
                target = int(self._store._ceiling_bytes * 0.8)
                evict_oldest(self._store, target, self._vector_store, self._document_store)
        except Exception as e:
            logger.debug("health check failed: %s", e)

        engine = self._evolution_engine
        if engine is not None and not engine._evaluating:
            try:
                signals = engine.compute_signals()
                engine.evaluate(signals)
            except Exception as e:
                logger.warning("evolution tick failed: %s", e)
