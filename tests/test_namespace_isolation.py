from supergraph import SuperGraph


def _ids(result):
    return [n["id"] for n in (result.data or [])]


def test_namespaced_node_excluded_from_default_view():
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    gs.execute('CREATE NODE "general1" kind = "memory" content = "general fact"')
    gs.execute('BIND NAMESPACE "intel:acme"')
    gs.execute('CREATE NODE "intel1" kind = "evidence" content = "secret intel"')
    gs.execute('DISCARD NAMESPACE')
    ids = _ids(gs.execute('NODES'))
    assert "general1" in ids
    assert "intel1" not in ids


def test_namespaced_node_visible_only_when_bound():
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    gs.execute('BIND NAMESPACE "intel:acme"')
    gs.execute('CREATE NODE "intel1" kind = "evidence" content = "x"')
    assert "intel1" in _ids(gs.execute('NODES'))
    gs.execute('DISCARD NAMESPACE')
    assert "intel1" not in _ids(gs.execute('NODES'))


def test_namespaces_are_mutually_isolated():
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    gs.execute('BIND NAMESPACE "a"')
    gs.execute('CREATE NODE "na" kind = "evidence" content = "a-fact"')
    gs.execute('DISCARD NAMESPACE')
    gs.execute('BIND NAMESPACE "b"')
    ids = _ids(gs.execute('NODES'))
    gs.execute('DISCARD NAMESPACE')
    assert "na" not in ids


def test_count_nodes_excludes_namespaced_by_default():
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    gs.execute('CREATE NODE "g" kind = "memory"')
    gs.execute('BIND NAMESPACE "intel"')
    gs.execute('CREATE NODE "i" kind = "evidence"')
    gs.execute('DISCARD NAMESPACE')
    assert gs.execute('COUNT NODES').count == 1


def test_lexical_retrieval_respects_namespace():
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    gs.execute('CREATE NODE "g1" kind = "memory" content = "public notes" DOCUMENT "public notes"')
    gs.execute('BIND NAMESPACE "intel"')
    gs.execute('CREATE NODE "i1" kind = "evidence" content = "secret breach" DOCUMENT "secret breach"')
    gs.execute('DISCARD NAMESPACE')
    assert "i1" not in _ids(gs.execute('LEXICAL SEARCH "secret breach" LIMIT 10'))
    gs.execute('BIND NAMESPACE "intel"')
    hit = _ids(gs.execute('LEXICAL SEARCH "secret breach" LIMIT 10'))
    gs.execute('DISCARD NAMESPACE')
    assert "i1" in hit


def test_edges_isolated_by_namespace_in_count():
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    gs.execute('CREATE NODE "g1" kind = "memory"')
    gs.execute('CREATE NODE "g2" kind = "memory"')
    gs.execute('CREATE EDGE "g1" -> "g2" kind = "rel"')
    gs.execute('BIND NAMESPACE "intel"')
    gs.execute('CREATE NODE "i1" kind = "evidence"')
    gs.execute('CREATE NODE "i2" kind = "evidence"')
    gs.execute('CREATE EDGE "i1" -> "i2" kind = "about"')
    gs.execute('DISCARD NAMESPACE')
    assert gs.execute('COUNT EDGES').count == 1
    assert gs.execute('COUNT EDGES', namespace="intel").count == 1


def test_namespace_and_context_are_mutually_exclusive():
    import pytest
    from supergraph.core.errors import SuperGraphError
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    gs.execute('BIND CONTEXT "c1"')
    with pytest.raises(SuperGraphError, match="(?i)namespace.*context|context.*namespace"):
        gs.execute('BIND NAMESPACE "intel"')
    gs.execute('DISCARD CONTEXT "c1"')
    gs.execute('BIND NAMESPACE "intel"')
    with pytest.raises(SuperGraphError, match="(?i)namespace.*context|context.*namespace"):
        gs.execute('BIND CONTEXT "c1"')
    gs.execute('DISCARD NAMESPACE')
