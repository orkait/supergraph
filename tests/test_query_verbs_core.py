import pytest

from supergraph import q
from supergraph.dsl.parser import parse


def _roundtrip(query_obj):
    dsl = query_obj.dsl()
    try:
        parse(dsl)
    except Exception as e:
        pytest.fail(f"parser rejected {dsl!r}: {e}")
    return dsl


class TestReadVerbs:
    def test_node(self):
        assert _roundtrip(q.node("mem:1")) == 'NODE "mem:1"'

    def test_node_with_document(self):
        assert _roundtrip(q.node("mem:1", with_document=True)) == 'NODE "mem:1" WITH DOCUMENT'

    def test_nodes_no_filter(self):
        dsl = _roundtrip(q.nodes())
        assert dsl == "NODES"

    def test_nodes_with_kind(self):
        dsl = _roundtrip(q.nodes(kind="memory"))
        assert 'kind = "memory"' in dsl

    def test_nodes_with_where_dict(self):
        dsl = _roundtrip(q.nodes(where={"kind": "memory", "importance__gt": 0.5}))
        assert 'kind = "memory"' in dsl
        assert "importance > 0.5" in dsl

    def test_nodes_with_limit(self):
        dsl = _roundtrip(q.nodes(kind="memory", limit=10))
        assert "LIMIT 10" in dsl

    def test_remember_basic(self):
        dsl = _roundtrip(q.remember("European history"))
        assert dsl == 'REMEMBER "European history"'

    def test_remember_full(self):
        dsl = _roundtrip(q.remember("x", limit=10, tokens=4000, at="2024-03"))
        assert 'REMEMBER "x"' in dsl
        assert 'AT "2024-03"' in dsl
        assert "LIMIT 10" in dsl
        assert "TOKENS 4000" in dsl

    def test_remember_empty_text_raises(self):
        with pytest.raises(ValueError):
            q.remember("")

    def test_recall(self):
        dsl = _roundtrip(q.recall("ent:paris", depth=2, limit=20))
        assert 'RECALL FROM "ent:paris"' in dsl
        assert "DEPTH 2" in dsl
        assert "LIMIT 20" in dsl

    def test_recall_negative_depth_raises(self):
        with pytest.raises(ValueError):
            q.recall("n1", depth=-1)

    def test_similar_text(self):
        dsl = _roundtrip(q.similar(text="capital city", limit=5))
        assert dsl == 'SIMILAR TO "capital city" LIMIT 5'

    def test_similar_node(self):
        dsl = _roundtrip(q.similar(node="mem:1", limit=5))
        assert 'SIMILAR TO NODE "mem:1"' in dsl

    def test_similar_vec(self):
        dsl = _roundtrip(q.similar(vec=[0.1, 0.2, 0.3], limit=5))
        assert "[0.1, 0.2, 0.3]" in dsl

    def test_similar_mutex_raises(self):
        with pytest.raises(ValueError, match="exactly one of"):
            q.similar(text="a", node="b")

    def test_similar_none_raises(self):
        with pytest.raises(ValueError, match="exactly one of"):
            q.similar()

    def test_lexical(self):
        dsl = _roundtrip(q.lexical("Eiffel Tower", limit=5))
        assert dsl == 'LEXICAL SEARCH "Eiffel Tower" LIMIT 5'

    def test_lexical_empty_raises(self):
        with pytest.raises(ValueError):
            q.lexical("")

    def test_edges(self):
        dsl = _roundtrip(q.edges("n1"))
        assert dsl == 'EDGES FROM "n1"'

    def test_count_nodes(self):
        dsl = _roundtrip(q.count_nodes(where={"kind": "memory"}))
        assert 'COUNT NODES WHERE kind = "memory"' == dsl


class TestWriteVerbs:
    def test_create_node_basic(self):
        dsl = _roundtrip(q.create_node("mem:1", kind="memory"))
        assert dsl == 'CREATE NODE "mem:1" kind = "memory"'

    def test_create_node_with_document(self):
        dsl = _roundtrip(q.create_node("mem:1", kind="memory", document="Paris"))
        assert 'CREATE NODE "mem:1"' in dsl
        assert 'DOCUMENT "Paris"' in dsl

    def test_create_node_with_fields(self):
        dsl = _roundtrip(q.create_node("mem:1", kind="memory", topic="travel", importance=0.9))
        assert 'topic = "travel"' in dsl
        assert "importance = 0.9" in dsl

    def test_create_node_clause_order_expires_before_document(self):
        dsl = _roundtrip(q.create_node("mem:1", kind="memory", expires_in="1h", document="x"))
        idx_expires = dsl.index("EXPIRES IN")
        idx_document = dsl.index("DOCUMENT")
        assert idx_expires < idx_document

    def test_create_node_none_document_omitted(self):
        dsl = _roundtrip(q.create_node("mem:1", kind="memory", document=None))
        assert "DOCUMENT" not in dsl


    def test_create_node_missing_kind_raises(self):
        with pytest.raises(TypeError):
            q.create_node("mem:1")

    def test_create_edge(self):
        dsl = _roundtrip(q.create_edge("a", "b", kind="calls"))
        assert dsl == 'CREATE EDGE "a" -> "b" kind = "calls"'

    def test_create_edge_with_fields(self):
        dsl = _roundtrip(q.create_edge("a", "b", kind="calls", weight=0.9))
        assert "weight = 0.9" in dsl

    def test_delete_node(self):
        dsl = _roundtrip(q.delete_node("mem:1"))
        assert dsl == 'DELETE NODE "mem:1"'


class TestCriticalEscape:

    def test_R3_kind_injection(self):
        malicious = 'mem"; DROP ALL; --'
        out = q.nodes(kind=malicious).dsl()
        assert r'\"' in out
        from supergraph.dsl.parser import parse
        parse(out)

    def test_R4_remember_quote_escape(self):
        out = q.remember('my "quoted" query').dsl()
        assert out == r'REMEMBER "my \"quoted\" query"'

    def test_W3_create_node_quote_escape(self):
        out = q.create_node("mem:1", kind="memory", document='text with "quotes"').dsl()
        assert r'DOCUMENT "text with \"quotes\""' in out

    def test_W4_create_node_none_document_omitted(self):
        out = q.create_node("mem:1", kind="memory", document=None).dsl()
        assert "DOCUMENT" not in out
        assert '"None"' not in out
