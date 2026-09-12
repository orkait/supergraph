import tempfile
from supergraph import SuperGraph


def test_remember_basic():
    gs = SuperGraph()
    gs.execute('CREATE NODE "fact1" kind = "fact" summary = "quantum entanglement is spooky"')
    gs.execute('CREATE NODE "fact2" kind = "fact" summary = "classical physics is deterministic"')
    gs.execute('CREATE NODE "fact3" kind = "fact" summary = "quantum computing uses qubits"')

    result = gs.execute('REMEMBER "quantum" LIMIT 5')
    assert result.kind == "nodes"
    gs.close()


def test_remember_with_persistence():
    with tempfile.TemporaryDirectory() as td:
        gs = SuperGraph(path=td)
        gs.execute('CREATE NODE "doc1" kind = "fact" summary = "photosynthesis converts light"')
        gs.execute('CREATE NODE "doc2" kind = "fact" summary = "mitochondria produces energy"')
        gs.execute('CREATE NODE "doc3" kind = "fact" summary = "chlorophyll absorbs light"')

        result = gs.execute('REMEMBER "light energy" LIMIT 5')
        assert result.kind == "nodes"
        gs.close()


def test_remember_returns_scores():
    gs = SuperGraph()
    gs.execute('CREATE NODE "a" kind = "test" summary = "hello world"')
    result = gs.execute('REMEMBER "hello" LIMIT 5')
    if result.data:
        node = result.data[0]
        assert "_remember_score" in node
        assert "_recency_score" in node
    gs.close()


def test_remember_with_where():
    gs = SuperGraph()
    gs.execute('CREATE NODE "a" kind = "fact" summary = "quantum physics"')
    gs.execute('CREATE NODE "b" kind = "opinion" summary = "quantum is weird"')
    result = gs.execute('REMEMBER "quantum" LIMIT 10 WHERE kind = "fact"')
    for node in result.data:
        assert node["kind"] == "fact"
    gs.close()


def test_remember_empty_store():
    gs = SuperGraph()
    result = gs.execute('REMEMBER "anything" LIMIT 5')
    assert result.kind == "nodes"
    assert result.data == []
    gs.close()


def test_remember_limit():
    gs = SuperGraph()
    for i in range(20):
        gs.execute(f'CREATE NODE "n{i}" kind = "test" summary = "test item {i}"')
    result = gs.execute('REMEMBER "test" LIMIT 3')
    assert len(result.data) <= 3
    gs.close()


def test_remember_at_without_event_column_warns():
    gs = SuperGraph()
    try:
        gs.execute('CREATE NODE "a" kind = "doc" text = "hello world"')
        r = gs.execute('REMEMBER "hello" AT "2024-01-01" LIMIT 5')
        warnings = r.meta.get("warnings", []) if r.meta else []
        assert any("__event_at__" in w for w in warnings), (
            f"Expected warning about missing __event_at__; got {warnings!r}"
        )
    finally:
        gs.close()


def test_remember_recall_count_persists_across_checkpoint(tmp_path):
    path = tmp_path / "gs"
    gs = SuperGraph(path=str(path))
    try:
        gs.execute('CREATE NODE "doc1" kind = "doc" text = "the quick brown fox" DOCUMENT "the quick brown fox"')
        r = gs.execute('REMEMBER "quick" LIMIT 5')
        assert r.count >= 1
        gs.checkpoint()
    finally:
        gs.close()

    gs2 = SuperGraph(path=str(path))
    try:
        cs = gs2._store
        n = cs._next_slot
        col = cs.columns.get_column("__recall_count__", n)
        assert col is not None, "__recall_count__ column lost across checkpoint"
        col_data, col_pres, _ = col
        assert int(col_data[col_pres].sum()) >= 1, "recall count value did not persist"
    finally:
        gs2.close()


def test_remember_reranker_error_surfaces_in_meta(tmp_path):
    class BrokenReranker:
        def score(self, q, docs):
            raise RuntimeError("rerank boom")

    gs = SuperGraph(path=str(tmp_path / "gs"))
    gs._executor._reranker = BrokenReranker()
    try:
        for i in range(20):
            gs.execute(
                f'CREATE NODE "d{i}" kind = "doc" text = "alpha beta {i}" '
                f'DOCUMENT "alpha beta {i}"'
            )
        r = gs.execute('REMEMBER "alpha" LIMIT 3')
        assert r.count == 3
        assert "reranker_error" in (r.meta or {}), (
            f"expected reranker_error in meta; got {r.meta!r}"
        )
    finally:
        gs.close()


def test_remember_nucleus_respects_visit_budget():
    gs = SuperGraph(nucleus_expansion=True, nucleus_hops=3,
                    nucleus_neighbors_per_hop=50,
                    nucleus_allowed_kinds=["chunk"])
    try:
        gs.execute('CREATE NODE "root" kind = "chunk" text = "seed chunk content"')
        for i in range(300):
            gs.execute(
                f'CREATE NODE "c{i}" kind = "chunk" '
                f'text = "chunk {i} body content must be long enough"'
            )
            src = "root" if i == 0 else f"c{i-1}"
            gs.execute(f'CREATE EDGE "{src}" -> "c{i}" kind = "next"')
        r = gs.execute('REMEMBER "seed" LIMIT 1')
        assert r.meta.get("nucleus_visits", 0) <= 150
    finally:
        gs.close()


def test_document_clause_populates_bm25_index():
    gs = SuperGraph(embedder=None)
    try:
        gs.execute('CREATE NODE "a" kind = "doc" text = "tag1" DOCUMENT "quantum entanglement"')
        gs.execute('CREATE NODE "b" kind = "doc" text = "tag2" DOCUMENT "classical mechanics"')
        r = gs.execute('REMEMBER "quantum" LIMIT 5')
        assert r.count >= 1, f"expected DOCUMENT content to surface via BM25, got {r.count} with meta={r.meta}"
        ids = {node["id"] for node in r.data}
        assert "a" in ids
    finally:
        gs.close()


def test_remember_empty_reports_diagnostic_reasons():
    gs = SuperGraph(embedder=None)
    try:
        gs.execute('CREATE NODE "a" kind = "doc" text = "just a column, no DOCUMENT"')
        r = gs.execute('REMEMBER "anything" LIMIT 5')
        reasons = (r.meta or {}).get("debug", {}).get("empty_result_reasons", [])
        assert r.count == 0
        assert reasons, f"expected diagnostic reasons; got meta={r.meta}"
        joined = " ".join(reasons).lower()
        assert "embedder" in joined or "vec" in joined or "bm25" in joined
    finally:
        gs.close()


def test_remember_empty_store_reports_no_nodes():
    gs = SuperGraph(embedder=None)
    try:
        r = gs.execute('REMEMBER "anything" LIMIT 5')
        reasons = (r.meta or {}).get("debug", {}).get("empty_result_reasons", [])
        assert any("no nodes" in r_.lower() for r_ in reasons), (
            f"expected 'no nodes' reason on empty store; got {reasons}"
        )
    finally:
        gs.close()
