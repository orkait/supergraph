from supergraph import SuperGraph


def _ids(r):
    return [n["id"] for n in (r.data or [])]


def test_per_query_namespace_scopes_read_no_global_leak_queued():
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False, queued=True)
    gs.execute('CREATE NODE "g1" kind = "memory"')
    gs.execute('UPSERT NODE "i1" kind = "evidence" __namespace__ = "intel"')

    scoped = _ids(gs.execute('NODES', namespace="intel"))
    assert scoped == ["i1"]

    after = _ids(gs.execute('NODES'))
    assert "g1" in after and "i1" not in after


def test_per_query_namespace_direct_mode():
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    gs.execute('UPSERT NODE "i1" kind = "evidence" __namespace__ = "intel"')
    gs.execute('CREATE NODE "g1" kind = "memory"')
    assert "i1" not in _ids(gs.execute('NODES'))
    assert "i1" in _ids(gs.execute('NODES', namespace="intel"))
    assert "i1" not in _ids(gs.execute('NODES'))


def test_per_query_namespace_write_then_scoped_read():
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False, queued=True)
    gs.execute('CREATE NODE "e1" kind = "evidence"', namespace="intel")
    assert "e1" not in _ids(gs.execute('NODES'))
    assert "e1" in _ids(gs.execute('NODES', namespace="intel"))
