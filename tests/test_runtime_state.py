
from supergraph import SuperGraph


def test_reset_memory_propagates_to_all_components(tmp_path):
    gs = SuperGraph(path=str(tmp_path))
    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED name')
    gs.execute('CREATE NODE "n1" name = "alpha" kind = "item"')
    gs.execute('CREATE NODE "n2" name = "beta" kind = "item"')

    assert gs._store.node_count == 2

    pre_store = gs._runtime.store
    gs.reset_memory()

    assert gs._runtime.store is not pre_store

    assert gs._executor.store is gs._runtime.store
    assert gs._sys_executor.store is gs._runtime.store
    assert gs._wal._store is gs._runtime.store
    assert gs._optimizer._store is gs._runtime.store

    assert gs._store.node_count == 0

    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED name')
    gs.execute('CREATE NODE "n3" name = "gamma" kind = "item"')
    assert gs._store.node_count == 1
    assert gs.execute('NODE "n3"').data is not None

    gs.close()


def test_lazy_vector_store_propagates_to_all_components(tmp_path):
    gs = SuperGraph(path=str(tmp_path), embedder=None)

    assert gs._runtime.vector_store is None
    assert gs._executor._vector_store is None
    assert gs._wal._vector_store is None

    gs.execute(
        'CREATE NODE "v1" kind = "item" name = "alpha" VECTOR [0.1, 0.2, 0.3, 0.4]'
    )

    vs = gs._runtime.vector_store
    assert vs is not None
    assert gs._executor._vector_store is vs
    assert gs._sys_executor._vector_store is vs
    assert gs._wal._vector_store is vs
    assert gs._optimizer._vector_store is vs

    gs.close()


def test_runtime_state_is_single_source_of_truth(tmp_path):
    gs = SuperGraph(path=str(tmp_path))

    runtime = gs._runtime
    assert gs._executor._runtime is runtime
    assert gs._sys_executor._runtime is runtime
    assert gs._wal._runtime is runtime
    assert gs._optimizer._runtime is runtime

    gs.close()


def test_rollback_vector_store_change_propagates(tmp_path):
    gs = SuperGraph(path=str(tmp_path))
    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED name')
    gs.execute(
        'CREATE NODE "v1" kind = "item" name = "alpha" VECTOR [0.1, 0.2, 0.3, 0.4]'
    )

    gs.execute('SYS SNAPSHOT "s1"')
    vs_at_snapshot = gs._runtime.vector_store
    assert vs_at_snapshot is not None

    gs.execute(
        'CREATE NODE "v2" kind = "item" name = "beta" VECTOR [0.5, 0.6, 0.7, 0.8]'
    )

    gs.execute('SYS ROLLBACK TO "s1"')

    vs_after_rollback = gs._runtime.vector_store
    assert gs._executor._vector_store is vs_after_rollback
    assert gs._sys_executor._vector_store is vs_after_rollback
    assert gs._wal._vector_store is vs_after_rollback
    assert gs._optimizer._vector_store is vs_after_rollback

    gs.close()
