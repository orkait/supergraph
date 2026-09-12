import threading
import time
from concurrent.futures import Future

from supergraph import SuperGraph
from supergraph.core.queue import CommandQueue


def test_queue_submit_returns_result():
    def fake_execute(query):
        return {"query": query}

    q = CommandQueue(fake_execute)
    result = q.submit("test")
    assert result == {"query": "test"}
    q.shutdown()


def test_queue_background_returns_future():
    def fake_execute(query):
        return {"query": query}

    q = CommandQueue(fake_execute)
    future = q.submit_background("bg_task")
    assert isinstance(future, Future)
    result = future.result(timeout=5)
    assert result == {"query": "bg_task"}
    q.shutdown()


def test_queue_priority_ordering():
    order = []
    gate = threading.Event()

    def slow_execute(query):
        if query == "blocker":
            gate.wait(timeout=5)
        order.append(query)
        return query

    q = CommandQueue(slow_execute)
    f_block = q.submit_background("blocker")
    time.sleep(0.05)

    f_interactive = q.submit_background("interactive_1")
    f_bg = q.submit_background("bg_1")
    interactive_result = []
    def submit_interactive():
        interactive_result.append(q.submit("interactive_2"))
    t = threading.Thread(target=submit_interactive)
    t.start()
    time.sleep(0.05)

    gate.set()
    f_block.result(timeout=5)
    t.join(timeout=5)

    assert order.index("interactive_2") < order.index("bg_1"), f"Order was: {order}"
    q.shutdown()


def test_queue_error_propagation():
    def failing_execute(query):
        raise ValueError("test error")

    q = CommandQueue(failing_execute)
    import pytest
    with pytest.raises(ValueError, match="test error"):
        q.submit("bad")
    with pytest.raises(ValueError, match="test error"):
        q.submit("also bad")
    q.shutdown()


def test_queue_shutdown_idempotent():
    def fake_execute(query):
        return query

    q = CommandQueue(fake_execute)
    q.shutdown()
    q.shutdown()


def test_queue_submit_after_shutdown_raises():
    def fake_execute(query):
        return query

    q = CommandQueue(fake_execute)
    q.shutdown()
    import pytest
    with pytest.raises(RuntimeError):
        q.submit("too late")


def test_supergraph_queued_execute():
    gs = SuperGraph(queued=True)
    result = gs.execute('CREATE NODE "test_t" kind = "item" name = "hello"')
    assert result.kind == "node"
    assert result.data["name"] == "hello"

    result2 = gs.execute('NODE "test_t"')
    assert result2.data["name"] == "hello"
    gs.close()


def test_supergraph_queued_background():
    gs = SuperGraph(queued=True)
    gs.execute('CREATE NODE "bg_test" kind = "item" name = "x"')
    future = gs.submit_background('NODE "bg_test"')
    assert isinstance(future, Future)
    result = future.result(timeout=5)
    assert result.data["name"] == "x"
    gs.close()


def test_supergraph_not_queued_rejects_background():
    gs = SuperGraph()
    import pytest
    with pytest.raises(RuntimeError, match="queued"):
        gs.submit_background('SYS STATS')
    gs.close()


def test_supergraph_concurrent_access():
    gs = SuperGraph(queued=True)
    errors = []
    results = []

    def worker(i):
        try:
            gs.execute(f'CREATE NODE "concurrent_{i}" kind = "item" val = {i}')
            r = gs.execute(f'NODE "concurrent_{i}"')
            results.append(r.data)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"Errors: {errors}"
    assert len(results) == 20
    gs.close()


def test_supergraph_default_not_queued():
    gs = SuperGraph()
    assert gs._queue is None
    result = gs.execute('CREATE NODE "noqueue" kind = "item"')
    assert result.kind == "node"
    gs.close()


def test_background_failure_logs_warning(caplog):
    import logging
    import time

    def failing_execute(query):
        if query == "fail_me":
            raise ValueError("intentional failure")
        return query

    q = CommandQueue(failing_execute)
    with caplog.at_level(logging.WARNING, logger="supergraph.core.queue"):
        future = q.submit_background("fail_me")
        try:
            future.result(timeout=5)
        except ValueError:
            pass
        time.sleep(0.05)
    assert any("background job failed" in r.message for r in caplog.records), \
        f"Expected warning log, got: {[r.message for r in caplog.records]}"
    q.shutdown()
