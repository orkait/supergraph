from supergraph import SuperGraph
import pytest

def test_sys_evict_limit():
    db = SuperGraph(ceiling_mb=100)
    # Ensure some protected schema to demonstrate they aren't evicted
    db.execute('SYS REGISTER NODE KIND "testing" REQUIRED field1:string')
    
    # Add nodes
    db.execute('CREATE NODE "x1" kind="testing" field1="a"')
    db.execute('CREATE NODE "x2" kind="testing" field1="b"')
    db.execute('CREATE NODE "x3" kind="testing" field1="c"')
    db.execute('CREATE NODE "x4" kind="testing" field1="d"')
    
    res = db.execute("SYS EVICT LIMIT 2")
    assert res.data["evicted"] == 2
    assert db.node_count == 2
    
def test_reset_store_semantics(tmp_path):
    # Test strict reset behaviors ensuring we wipe, but allow recreation seamlessly
    # Ensure VectorStore is fully wiped, SQLite tables wiped, but schema and DB reconnects correctly.
    db_dir = tmp_path / "test_reset"
    db_dir.mkdir()
    
    db = SuperGraph(path=str(db_dir))
    db.execute('SYS REGISTER NODE KIND "animal" REQUIRED name:string')
    db.execute('CREATE NODE "dog" kind="animal" name="fido"')
    db.execute('CREATE NODE "cat" kind="animal" name="felix"')
    
    assert db.node_count == 2
    db.set_script('CREATE NODE "parrot" kind="animal" name="polly"')
    
    # Do reset
    res = db.reset_store(preserve_config=True)
    assert res.kind == "ok"
    
    # Store should be empty
    assert db.node_count == 0
    
    # Script should be preserved
    assert db.get_script() == 'CREATE NODE "parrot" kind="animal" name="polly"'
    
    # Can still run things
    db.execute('CREATE NODE "mouse" kind="animal" name="mickey"')
    assert db.node_count == 1
    
def test_config_semantics(tmp_path):
    db_dir = tmp_path / "test_config"
    db_dir.mkdir()
    
    db = SuperGraph(path=str(db_dir))
    
    res = db.get_runtime_config()
    assert res.data["core"]["ceiling_mb"] == 256
    
    # Update runtime config
    res = db.update_runtime_config({"ceiling_mb": 512, "cost_threshold": 2000})
    assert db.ceiling_mb == 512
    assert db.cost_threshold == 2000
    
    # Update persisted config
    res = db.update_persisted_config({"ceiling_mb": 1024})
    
    # Runtime should still be 512
    assert db.ceiling_mb == 512
    
    # But read persisted should show 1024
    res = db.get_persisted_config()
    assert res.data["core"]["ceiling_mb"] == 1024
    
    # New instance loading that config should start with 1024
    # (close the first; single-owner lock prevents concurrent opens on the
    # same path)
    db.close()
    db2 = SuperGraph(path=str(db_dir))
    assert db2.ceiling_mb == 1024
    db2.close()


# ============================================================
# Hardening: eviction, reset, session, rollback contracts
# ============================================================

def test_sys_evict_without_limit():
    """SYS EVICT without LIMIT should use evict_oldest to target bytes.
    Previously this crashed with NameError: evict_by_bytes."""
    db = SuperGraph(ceiling_mb=100, embedder=None)
    db.execute('CREATE NODE "e1" kind="test" field1="a"')
    db.execute('CREATE NODE "e2" kind="test" field1="b"')
    db.execute('CREATE NODE "e3" kind="test" field1="c"')

    res = db.execute("SYS EVICT")
    assert "evicted" in res.data
    assert isinstance(res.data["evicted"], int)


def test_reset_memory_then_write():
    """After reset_memory, creating new nodes should work without errors.
    Tests that executor, optimizer, and WAL references are all synced."""
    db = SuperGraph(ceiling_mb=100, embedder=None)
    db.execute('CREATE NODE "a1" kind="test" name="before"')
    assert db.node_count == 1

    db.reset_memory()
    assert db.node_count == 0

    db.execute('CREATE NODE "b1" kind="test" name="after"')
    assert db.node_count == 1
    node = db.execute('NODE "b1"')
    assert node.data is not None
    assert node.data["name"] == "after"


def test_reset_memory_syncs_optimizer():
    """reset_memory must update the optimizer's store reference.
    Without this fix, auto-optimize would operate on the old (stale) store."""
    db = SuperGraph(ceiling_mb=100, embedder=None)
    db.execute('CREATE NODE "z1" kind="test" x="1"')

    old_store = db._store
    db.reset_memory()

    assert db._optimizer._store is db._store
    assert db._optimizer._store is not old_store
    assert db._optimizer._vector_store is db._vector_store


def test_reset_session_clears_dirty_and_trace():
    """reset_session clears _embedder_dirty and _active_trace."""
    db = SuperGraph(ceiling_mb=100, embedder=None)
    db._embedder_dirty = True
    db.bind_trace("test-session")
    assert db._active_trace == "test-session"

    res = db.reset_session()
    assert res.kind == "ok"
    assert db._embedder_dirty is False
    assert db._active_trace is None


def test_scheduler_emergency_eviction_arg_order():
    """Verify the scheduler's evict_oldest call has correct argument order.
    Previously it passed vector_store as target_bytes (positional), causing
    TypeError: got multiple values for argument 'target_bytes'."""
    from supergraph.core.optimizer import evict_oldest
    import inspect

    sig = inspect.signature(evict_oldest)
    params = list(sig.parameters.keys())
    assert params[0] == "store"
    assert params[1] == "target_bytes"
    assert params[2] == "vector_store"
    assert params[3] == "document_store"

    # Functional test: the scheduler's code path should not raise
    db = SuperGraph(ceiling_mb=1, embedder=None)
    for i in range(10):
        db.execute(f'CREATE NODE "s{i}" kind="test" x="{i}"')

    # Manually trigger the scheduler's health check logic
    try:
        db._optimizer._check_health()
    except Exception as e:
        pytest.fail(f"Scheduler health check raised: {e}")


def test_rollback_syncs_all_references(tmp_path):
    """After SYS ROLLBACK, all components (executor, WAL, optimizer)
    should reference the same vector store."""
    db_dir = tmp_path / "rollback_test"
    db_dir.mkdir()
    db = SuperGraph(path=str(db_dir), embedder=None)

    db.execute('CREATE NODE "r1" kind="test" name="snap"')
    db.execute('SYS SNAPSHOT "s1"')
    db.execute('CREATE NODE "r2" kind="test" name="post"')
    assert db.node_count == 2

    db.execute('SYS ROLLBACK TO "s1"')
    assert db.node_count == 1

    # WAL and optimizer should reference the same vector store as SuperGraph
    assert db._wal._vector_store is db._vector_store
    assert db._optimizer._vector_store is db._vector_store
    assert db._executor._vector_store is db._vector_store
    db.close()

