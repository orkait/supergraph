
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

    def __init__(self, execute_fn: Callable[[str], Any]):
        self._execute_fn = execute_fn
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = 0
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True, name="supergraph-worker")
        self._running = True
        self._worker.start()

    def _next_seq_locked(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    def submit(self, query: str, namespace: str | None = None) -> Any:
        future: Future = Future()
        with self._lock:
            if not self._running:
                raise RuntimeError("CommandQueue is shut down")
            self._queue.put((INTERACTIVE, self._next_seq_locked(), query, future, namespace))
        return future.result()

    def submit_background(self, query: str, namespace: str | None = None) -> Future:
        future: Future = Future()
        future.add_done_callback(lambda f: self._on_background_done(f, query))
        with self._lock:
            if not self._running:
                raise RuntimeError("CommandQueue is shut down")
            self._queue.put((BACKGROUND, self._next_seq_locked(), query, future, namespace))
        return future

    @staticmethod
    def _on_background_done(future: Future, query: str) -> None:
        exc = future.exception()
        if exc is not None:
            logger.warning("background job failed: %s - %s: %s", query, type(exc).__name__, exc)

    def shutdown(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._queue.put((999, 0, _SHUTDOWN, None, None))
        self._worker.join(timeout=5)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            priority, seq, query, future, namespace = item
            if query is _SHUTDOWN:
                break
            try:
                if namespace is None:
                    result = self._execute_fn(query)
                else:
                    result = self._execute_fn(query, namespace=namespace)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

    @property
    def pending(self) -> int:
        return self._queue.qsize()
