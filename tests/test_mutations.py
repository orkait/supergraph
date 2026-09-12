import pytest
from supergraph import SuperGraph


def test_batch_vector_rolled_back_on_failure():
    gs = SuperGraph(embedder=None)
    try:
        gs.execute('CREATE NODE "seed" kind = "doc" text = "hello"')
        assert gs._vector_store is None or gs._vector_store.count() == 0

        with pytest.raises(Exception):
            gs.execute(
                'BEGIN\n'
                'CREATE NODE "x" kind = "doc" text = "a" VECTOR [0.1, 0.2, 0.3]\n'
                'CREATE NODE "x" kind = "doc" text = "b"\n'
                'COMMIT'
            )

        assert gs.execute('NODE "x"').data is None
        vs = gs._vector_store
        assert vs is None or vs.count() == 0
    finally:
        gs.close()


def test_auto_id_width_is_16_hex():
    gs = SuperGraph()
    try:
        r = gs.execute('CREATE NODE AUTO kind = "k" v = 1')
        nid = r.data["id"]
        assert len(nid) == 16, f"expected 16 hex chars, got {len(nid)}: {nid!r}"
        assert all(c in "0123456789abcdef" for c in nid)
    finally:
        gs.close()


def test_batch_rollback_removes_multiple_pending_vectors():
    gs = SuperGraph(embedder=None)
    try:
        assert gs._vector_store is None or gs._vector_store.count() == 0

        with pytest.raises(Exception):
            gs.execute(
                'BEGIN\n'
                'CREATE NODE "a" kind = "doc" text = "x" VECTOR [0.1, 0.2, 0.3]\n'
                'CREATE NODE "b" kind = "doc" text = "y" VECTOR [0.4, 0.5, 0.6]\n'
                'CREATE NODE "c" kind = "doc" text = "z" VECTOR [0.7, 0.8, 0.9]\n'
                'CREATE NODE "a" kind = "doc" text = "dup"\n'
                'COMMIT'
            )

        for nid in ("a", "b", "c"):
            assert gs.execute(f'NODE "{nid}"').data is None, f"{nid} should be rolled back"
        vs = gs._vector_store
        assert vs is None or vs.count() == 0, "no vectors must survive rollback"
    finally:
        gs.close()


def test_batch_disabled_rollback_leaves_committed_side_effects():
    gs = SuperGraph(enable_rollback=False, embedder=None)
    try:
        with pytest.raises(Exception):
            gs.execute(
                'BEGIN\n'
                'CREATE NODE "aa" kind = "doc" text = "x"\n'
                'CREATE EDGE "aa" -> "missing_target" kind = "r"\n'
                'COMMIT'
            )
        assert gs.execute('NODE "aa"').data is not None, (
            "rollback disabled -> pre-failure inserts must persist"
        )
    finally:
        gs.close()
