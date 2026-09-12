"""Write-Ahead Log management for supergraph persistence."""

import time
import logging

logger = logging.getLogger(__name__)
_event_logger = logging.getLogger("supergraph.events")

from supergraph.core.errors import SuperGraphError, NodeExists
from supergraph.core.runtime import RuntimeState
from supergraph.dsl.parser import parse
from supergraph.persistence.serializer import checkpoint as _checkpoint_fn


class WALManager:
    """Manages WAL append, replay, auto-checkpoint, and query log rotation."""

    def __init__(self, runtime: RuntimeState, executor,
                 wal_hard_limit: int, auto_checkpoint_threshold: int,
                 log_retention_days: int, strict_recovery: bool = False):
        self._runtime = runtime
        self._executor = executor
        self._wal_hard_limit = wal_hard_limit
        self._auto_checkpoint_threshold = auto_checkpoint_threshold
        self._log_retention_days = log_retention_days
        self._strict_recovery = strict_recovery
        self._replay_errors: list[dict] = []
        self._query_log_max_rows = 200_000

    @property
    def _conn(self):
        return self._runtime.conn

    @property
    def _store(self):
        return self._runtime.store

    @property
    def _schema(self):
        return self._runtime.schema

    @property
    def _vector_store(self):
        return self._runtime.vector_store

    @property
    def pending_count(self) -> int:
        """Number of WAL entries not yet checkpointed."""
        if self._conn is None:
            return 0
        row = self._conn.execute("SELECT COUNT(*) FROM wal").fetchone()
        return row[0] if row else 0

    def append(self, statement: str) -> None:
        conn = self._conn
        if conn is None:
            return
        row = conn.execute("SELECT COUNT(*) FROM wal").fetchone()
        if row and row[0] >= self._wal_hard_limit:
            try:
                self.checkpoint()
            except Exception as e:
                logger.debug("WAL checkpoint before hard-limit failed: %s", e, exc_info=True)
            row = conn.execute("SELECT COUNT(*) FROM wal").fetchone()
            if row and row[0] >= self._wal_hard_limit:
                raise SuperGraphError(
                    f"WAL exceeds hard limit ({self._wal_hard_limit} entries). "
                    "Checkpoint failed - check disk space."
                )
        conn.execute(
            "INSERT INTO wal (timestamp, statement) VALUES (?, ?)",
            (time.time(), statement)
        )
        conn.commit()

    def replay(self) -> None:
        conn = self._conn
        if conn is None:
            return
        rows = conn.execute("SELECT seq, statement FROM wal ORDER BY seq").fetchall()
        if not rows:
            return
        self._replay_errors.clear()

        # Wrap the entire replay loop in a single sqlite transaction. Pre-fix
        # each failed-statement branch committed independently (bug #5). If
        # the process crashed mid-replay, some WAL entries were deleted
        # while others were retried on next boot — state drift was possible
        # when later-in-replay statements depended on earlier ones. With a
        # single enclosing transaction, a crash leaves the WAL exactly as it
        # was before replay started, and the next boot re-runs from a clean
        # starting point.
        #
        # We use an IMMEDIATE transaction rather than DEFERRED so the lock
        # is acquired up-front; DEFERRED would silently upgrade on first
        # write and risk SQLITE_BUSY under concurrent access.
        conn.execute("BEGIN IMMEDIATE")
        try:
            for seq, statement in rows:
                try:
                    ast = parse(statement)
                    self._executor.execute(ast)
                except NodeExists:
                    # The node already exists. Re-applying the CREATE NODE is
                    # a no-op; the right thing is to drop the WAL entry so the
                    # next boot does not retry it indefinitely (bug #4).
                    # Pre-fix this branch just ``continue``d, leaving the
                    # offending entry in the WAL where it would fail the
                    # same way on every subsequent replay.
                    try:
                        conn.execute("DELETE FROM wal WHERE seq = ?", (seq,))
                    except Exception as del_e:
                        logger.error(
                            "Failed to delete NodeExists WAL entry seq=%s: %s",
                            seq, del_e,
                        )
                    continue
                except Exception as e:
                    if self._strict_recovery:
                        logger.error(
                            "Fatal error during WAL replay (strict_recovery=True): %s", e,
                        )
                        # Undo any partial replay before re-raising; the outer
                        # caller expects "either fully replayed or untouched".
                        conn.execute("ROLLBACK")
                        raise SuperGraphError(
                            f"Fatal recovery error on statement: {statement}"
                        ) from e

                    err = {
                        "statement": statement[:500],
                        "error_type": type(e).__name__,
                        "error": str(e)[:500],
                    }
                    if len(self._replay_errors) < 50:
                        self._replay_errors.append(err)
                    logger.warning(
                        "WAL replay statement failed: %s: %s - %s",
                        err["error_type"], err["error"], err["statement"],
                    )

                    # Dead-letter queue + delete the failing WAL entry.
                    # Both ops participate in the outer transaction; they
                    # commit together at the end.
                    try:
                        conn.execute(
                            "INSERT INTO failed_wal_entries (timestamp, statement, error_msg) VALUES (?, ?, ?)",
                            (time.time(), statement, err["error"])
                        )
                    except Exception as dlq_e:
                        logger.error("Failed to log to dead letter queue: %s", dlq_e)
                    try:
                        conn.execute("DELETE FROM wal WHERE seq = ?", (seq,))
                    except Exception as del_e:
                        logger.error(
                            "Failed to delete failed WAL entry seq=%s: %s",
                            seq, del_e,
                        )
            # Commit the whole replay atomically. Only reached if no branch
            # raised and triggered the ROLLBACK above.
            conn.execute("COMMIT")
        except Exception:
            # Any uncaught exception (strict-recovery fatal path re-raised,
            # or sqlite IO error) leaves the WAL untouched.
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

        if self._replay_errors:
            logger.warning(
                "WAL replay completed with %d error(s). See SYS FAILED QUERIES for details.",
                len(self._replay_errors),
            )
        if rows:
            _checkpoint_fn(self._store, self._schema, conn, force=True)

    @property
    def replay_error_count(self) -> int:
        return len(self._replay_errors)

    @property
    def replay_errors(self) -> list[dict]:
        return list(self._replay_errors)

    def checkpoint(self, vector_store=None) -> None:
        conn = self._conn
        if conn is None:
            return
        if self._vector_store is not None:
            self._store.vectors = self._vector_store
        _checkpoint_fn(self._store, self._schema, conn)
        self._store.reset_dirty_flags()

    def maybe_auto_checkpoint(self) -> None:
        conn = self._conn
        if conn is None:
            return
        row = conn.execute("SELECT COUNT(*) FROM wal").fetchone()
        if row and row[0] > self._auto_checkpoint_threshold:
            self.checkpoint()
        self._rotate_query_log()

    def log_query(self, query: str, elapsed_us: int, result_count: int, error: str | None,
                  tag: str | None = None, trace_id: str | None = None,
                  source: str = "user", phase: str | None = None) -> None:
        conn = self._conn
        if conn is None:
            return
        try:
            conn.execute(
                "INSERT INTO query_log (timestamp, query, elapsed_us, result_count, error, tag, trace_id, source, phase) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (time.time(), query, elapsed_us, result_count, error, tag, trace_id, source, phase)
            )
            conn.commit()
        except Exception as e:
            logger.debug("query log insert failed: %s", e, exc_info=True)

    def emit_event(self, query: str, elapsed_us: int, result_count: int, error: str | None,
                   tag: str | None = None, trace_id: str | None = None,
                   source: str = "user", phase: str | None = None) -> None:
        """Emit structured log event via supergraph.events logger."""
        _event_logger.info(
            "%s [%s] %dus %d results",
            tag or "unknown", source, elapsed_us, result_count,
            extra={
                "gs_query": query,
                "gs_tag": tag,
                "gs_source": source,
                "gs_trace_id": trace_id,
                "gs_phase": phase,
                "gs_elapsed_us": elapsed_us,
                "gs_result_count": result_count,
                "gs_error": error,
            },
        )

    def _rotate_query_log(self) -> None:
        conn = self._conn
        if conn is None:
            return
        try:
            cutoff = time.time() - self._log_retention_days * 86400
            conn.execute("DELETE FROM query_log WHERE timestamp < ?", (cutoff,))
            row = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()
            count = row[0] if row else 0
            if count > self._query_log_max_rows:
                conn.execute(
                    "DELETE FROM query_log WHERE id IN ("
                    "  SELECT id FROM query_log ORDER BY timestamp ASC LIMIT ?"
                    ")",
                    (count - self._query_log_max_rows,),
                )
            conn.commit()
        except Exception as e:
            logger.debug("query log rotation failed: %s", e, exc_info=True)
