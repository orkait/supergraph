"""Single-writer command queue for thread-safe SuperGraph access.

All execute() calls are serialized through a PriorityQueue drained by
a dedicated daemon worker thread. Two priority levels ensure interactive
queries complete before background refinement jobs.
"""

from __future__ import annotations

import logging
import threading
import queue
from concurrent.futures import Future
from typing import Callable, Any

logger = logging.getLogger(__name__)

INTERACTIVE = 0
BACKGROUND = 1

_SHUTDOWN = object()


class CommandQueue:
    """Priority queue with dedicated worker thread for serialized execution."""

    def __init__(self, execute_fn: Callable[[str], Any]):
        self._execute_fn = execute_fn
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = 0
        # Single lock protects: the _running flag, the _seq counter, and
        # the atomic check-and-enqueue critical section in submit*().
        # Pre-fix, submit() checked _running and then put() as two separate
        # operations, so shutdown() could interleave and leave a future
        # sitting on a queue whose worker had already exited — blocking
        # the caller's .result() forever (bug #103).
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True, name="supergraph-worker")
        self._running = True
        self._worker.start()

    def _next_seq_locked(self) -> int:
        """Caller must hold self._lock."""
        seq = self._seq
        self._seq += 1
        return seq

    def submit(self, query: str, namespace: str | None = None) -> Any:
        """Submit interactive query, block until result.

        ``namespace`` is applied for the duration of this query only, inside
        the worker thread (serialized), so a scoped read never leaks global
        _active_namespace state to other queries.
        """
        future: Future = Future()
        # Atomic: _running check + seq allocation + put. If shutdown()
        # is waiting for the lock, it will run AFTER this put completes,
        # so the worker is guaranteed to see this item before it sees
        # the _SHUTDOWN sentinel.
        with self._lock:
            if not self._running:
                raise RuntimeError("CommandQueue is shut down")
            self._queue.put((INTERACTIVE, self._next_seq_locked(), query, future, namespace))
        return future.result()

    def submit_background(self, query: str, namespace: str | None = None) -> Future:
        """Submit background query, return Future immediately.

        Failed background jobs are logged at WARNING level even if
        the caller never calls .result() on the returned Future.
        """
        future: Future = Future()
        future.add_done_callback(lambda f: self._on_background_done(f, query))
        with self._lock:
            if not self._running:
                raise RuntimeError("CommandQueue is shut down")
            self._queue.put((BACKGROUND, self._next_seq_locked(), query, future, namespace))
        return future

    @staticmethod
    def _on_background_done(future: Future, query: str) -> None:
        """Log failed background jobs."""
        exc = future.exception()
        if exc is not None:
            logger.warning("background job failed: %s - %s: %s", query, type(exc).__name__, exc)

    def shutdown(self) -> None:
        """Stop the worker thread. Idempotent."""
        # Acquire lock before flipping _running so no submit() can race
        # in between our check and our _SHUTDOWN sentinel put. Any
        # submit() already inside the lock will complete its put first;
        # any submit() that acquires the lock after us sees
        # _running=False and raises RuntimeError.
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._queue.put((999, 0, _SHUTDOWN, None, None))
        self._worker.join(timeout=5)

    def _run(self) -> None:
        """Worker loop: drain queue, execute, set results."""
        while True:
            item = self._queue.get()
            priority, seq, query, future, namespace = item
            if query is _SHUTDOWN:
                break
            try:
                # Only thread namespace when set, so generic execute_fn(query)
                # callers (and tests) keep working unchanged.
                if namespace is None:
                    result = self._execute_fn(query)
                else:
                    result = self._execute_fn(query, namespace=namespace)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

    @property
    def pending(self) -> int:
        """Approximate number of pending items."""
        return self._queue.qsize()
