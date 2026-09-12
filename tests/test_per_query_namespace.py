"""Per-query namespace scoping: a single read can be scoped to a namespace
WITHOUT leaving global _active_namespace state set. This is what lets a harness
(siphon) on a shared, queued single-writer server read its isolated corpus
without breaking every other consumer's reads (the global-BIND hazard)."""
from supergraph import SuperGraph


def _ids(r):
    return [n["id"] for n in (r.data or [])]


def test_per_query_namespace_scopes_read_no_global_leak_queued():
    # queued=True mirrors the HTTP server (single-writer worker thread)
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False, queued=True)
    gs.execute('CREATE NODE "g1" kind = "memory"')
    gs.execute('UPSERT NODE "i1" kind = "evidence" __namespace__ = "intel"')

    scoped = _ids(gs.execute('NODES', namespace="intel"))
    assert scoped == ["i1"]                       # per-query scope sees only intel

    # CRITICAL: the very next normal read must NOT be globally filtered
    after = _ids(gs.execute('NODES'))
    assert "g1" in after and "i1" not in after    # general view clean, server not stuck in 'intel'


def test_per_query_namespace_direct_mode():
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    gs.execute('UPSERT NODE "i1" kind = "evidence" __namespace__ = "intel"')
    gs.execute('CREATE NODE "g1" kind = "memory"')
    assert "i1" not in _ids(gs.execute('NODES'))                 # excluded from default
    assert "i1" in _ids(gs.execute('NODES', namespace="intel"))  # visible per-query
    assert "i1" not in _ids(gs.execute('NODES'))                 # restored, no leak


def test_per_query_namespace_write_then_scoped_read():
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False, queued=True)
    # a write under per-query namespace tags the node
    gs.execute('CREATE NODE "e1" kind = "evidence"', namespace="intel")
    assert "e1" not in _ids(gs.execute('NODES'))                 # not in general view
    assert "e1" in _ids(gs.execute('NODES', namespace="intel"))  # in its namespace
