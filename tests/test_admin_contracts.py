from supergraph import SuperGraph
import pytest

def test_sys_evict_limit():
    db = SuperGraph(ceiling_mb=100)
    db.execute('SYS REGISTER NODE KIND "testing" REQUIRED field1:string')
    
    db.execute('CREATE NODE "x1" kind="testing" field1="a"')
    db.execute('CREATE NODE "x2" kind="testing" field1="b"')
    db.execute('CREATE NODE "x3" kind="testing" field1="c"')
    db.execute('CREATE NODE "x4" kind="testing" field1="d"')
    
    res = db.execute("SYS EVICT LIMIT 2")
    assert res.data["evicted"] == 2
    assert db.node_count == 2
    
def test_reset_store_semantics(tmp_path):
    db_dir = tmp_path / "test_reset"
    db_dir.mkdir()
    
    db = SuperGraph(path=str(db_dir))
    db.execute('SYS REGISTER NODE KIND "animal" REQUIRED name:string')
    db.execute('CREATE NODE "dog" kind="animal" name="fido"')
    db.execute('CREATE NODE "cat" kind="animal" name="felix"')
    
    assert db.node_count == 2
    db.set_script('CREATE NODE "parrot" kind="animal" name="polly"')
    
    res = db.reset_store(preserve_config=True)
    assert res.kind == "ok"
    
    assert db.node_count == 0
    
    assert db.get_script() == 'CREATE NODE "parrot" kind="animal" name="polly"'
    
    db.execute('CREATE NODE "mouse" kind="animal" name="mickey"')
    assert db.node_count == 1
    
def test_config_semantics(tmp_path):
    db_dir = tmp_path / "test_config"
    db_dir.mkdir()
    
    db = SuperGraph(path=str(db_dir))
    
    res = db.get_runtime_config()
    assert res.data["core"]["ceiling_mb"] == 256
    
    res = db.update_runtime_config({"ceiling_mb": 512, "cost_threshold": 2000})
    assert db.ceiling_mb == 512
    assert db.cost_threshold == 2000
    
    res = db.update_persisted_config({"ceiling_mb": 1024})
    
    assert db.ceiling_mb == 512
    
    res = db.get_persisted_config()
    assert res.data["core"]["ceiling_mb"] == 1024
    
    db.close()
    db2 = SuperGraph(path=str(db_dir))
    assert db2.ceiling_mb == 1024
    db2.close()


def test_sys_evict_without_limit():
    db = SuperGraph(ceiling_mb=100, embedder=None)
    db.execute('CREATE NODE "e1" kind="test" field1="a"')
    db.execute('CREATE NODE "e2" kind="test" field1="b"')
    db.execute('CREATE NODE "e3" kind="test" field1="c"')

    res = db.execute("SYS EVICT")
    assert "evicted" in res.data
    assert isinstance(res.data["evicted"], int)


def test_reset_memory_then_write():
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
    db = SuperGraph(ceiling_mb=100, embedder=None)
    db.execute('CREATE NODE "z1" kind="test" x="1"')

    old_store = db._store
    db.reset_memory()

    assert db._optimizer._store is db._store
    assert db._optimizer._store is not old_store
    assert db._optimizer._vector_store is db._vector_store


def test_reset_session_clears_dirty_and_trace():
    db = SuperGraph(ceiling_mb=100, embedder=None)
    db._embedder_dirty = True
    db.bind_trace("test-session")
    assert db._active_trace == "test-session"

    res = db.reset_session()
    assert res.kind == "ok"
    assert db._embedder_dirty is False
    assert db._active_trace is None


def test_scheduler_emergency_eviction_arg_order():
    from supergraph.core.optimizer import evict_oldest
    import inspect

    sig = inspect.signature(evict_oldest)
    params = list(sig.parameters.keys())
    assert params[0] == "store"
    assert params[1] == "target_bytes"
    assert params[2] == "vector_store"
    assert params[3] == "document_store"

    db = SuperGraph(ceiling_mb=1, embedder=None)
    for i in range(10):
        db.execute(f'CREATE NODE "s{i}" kind="test" x="{i}"')

    try:
        db._optimizer._check_health()
    except Exception as e:
        pytest.fail(f"Scheduler health check raised: {e}")


def test_rollback_syncs_all_references(tmp_path):
    db_dir = tmp_path / "rollback_test"
    db_dir.mkdir()
    db = SuperGraph(path=str(db_dir), embedder=None)

    db.execute('CREATE NODE "r1" kind="test" name="snap"')
    db.execute('SYS SNAPSHOT "s1"')
    db.execute('CREATE NODE "r2" kind="test" name="post"')
    assert db.node_count == 2

    db.execute('SYS ROLLBACK TO "s1"')
    assert db.node_count == 1

    assert db._wal._vector_store is db._vector_store
    assert db._optimizer._vector_store is db._vector_store
    assert db._executor._vector_store is db._vector_store
    db.close()

