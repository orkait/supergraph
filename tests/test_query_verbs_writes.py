import pytest

from supergraph import q, F
from supergraph.dsl.parser import parse


def _roundtrip(query_obj):
    dsl = query_obj.dsl()
    try:
        parse(dsl)
    except Exception as e:
        pytest.fail(f"parser rejected {dsl!r}: {e}")
    return dsl


class TestUpdateNode:
    def test_basic(self):
        assert _roundtrip(q.update_node("m1", name="new")) == 'UPDATE NODE "m1" SET name = "new"'

    def test_multiple_fields(self):
        dsl = _roundtrip(q.update_node("m1", name="new", importance=0.9))
        assert 'name = "new"' in dsl
        assert "importance = 0.9" in dsl

    def test_no_fields_raises(self):
        with pytest.raises(ValueError):
            q.update_node("m1")


class TestUpsertNode:
    def test_basic(self):
        dsl = _roundtrip(q.upsert_node("m1", kind="memory", topic="x"))
        assert dsl == 'UPSERT NODE "m1" kind = "memory" topic = "x"'

    def test_with_event_at(self):
        dsl = _roundtrip(q.upsert_node("m1", kind="memory", event_at="2024-03-15"))
        assert 'EVENT_AT "2024-03-15"' in dsl


class TestDeleteNodes:
    def test_basic(self):
        dsl = _roundtrip(q.delete_nodes(where=F.eq("kind", "test")))
        assert dsl == 'DELETE NODES WHERE kind = "test"'

    def test_no_where_raises(self):
        with pytest.raises(ValueError):
            q.delete_nodes(where=None)


class TestUpdateNodes:
    def test_basic(self):
        dsl = _roundtrip(q.update_nodes(where=F.eq("kind", "fact"), set={"confidence": 0.5}))
        assert "UPDATE NODES" in dsl
        assert 'WHERE kind = "fact"' in dsl
        assert "SET" in dsl
        assert "confidence = 0.5" in dsl

    def test_no_set_raises(self):
        with pytest.raises(ValueError):
            q.update_nodes(where=F.eq("x", 1), set={})


class TestUpdateEdge:
    def test_basic(self):
        dsl = _roundtrip(q.update_edge("a", "b", set={"weight": 0.9}))
        assert dsl == 'UPDATE EDGE "a" -> "b" SET weight = 0.9'

    def test_with_where(self):
        dsl = _roundtrip(q.update_edge("a", "b", set={"weight": 0.9}, where=F.eq("kind", "next")))
        assert "WHERE" in dsl


class TestDeleteEdge:
    def test_basic(self):
        assert _roundtrip(q.delete_edge("a", "b")) == 'DELETE EDGE "a" -> "b"'

    def test_with_where(self):
        dsl = _roundtrip(q.delete_edge("a", "b", where=F.eq("kind", "next")))
        assert "WHERE" in dsl


class TestDeleteEdges:
    def test_from(self):
        assert _roundtrip(q.delete_edges("n1", direction="FROM")) == 'DELETE EDGES FROM "n1"'

    def test_to(self):
        assert _roundtrip(q.delete_edges("n1", direction="TO")) == 'DELETE EDGES TO "n1"'

    def test_with_where(self):
        dsl = _roundtrip(q.delete_edges("n1", direction="FROM", where=F.eq("kind", "next")))
        assert "WHERE" in dsl


class TestIncrement:
    def test_basic(self):
        assert _roundtrip(q.increment("m1", "hits", by=1)) == 'INCREMENT NODE "m1" hits BY 1'

    def test_negative_by(self):
        assert _roundtrip(q.increment("m1", "hits", by=-2)) == 'INCREMENT NODE "m1" hits BY -2'


class TestAssert:
    def test_basic(self):
        dsl = _roundtrip(q.assert_("f1", kind="fact", value=42))
        assert dsl == 'ASSERT "f1" kind = "fact" value = 42'

    def test_with_confidence_source(self):
        dsl = _roundtrip(q.assert_("f1", kind="fact", value=42, confidence=0.9, source="tool"))
        assert "CONFIDENCE 0.9" in dsl
        assert 'SOURCE "tool"' in dsl

    def test_with_event_at(self):
        dsl = _roundtrip(q.assert_("f1", kind="fact", event_at="2024-01"))
        assert 'EVENT_AT "2024-01"' in dsl

    def test_missing_kind_raises(self):
        with pytest.raises(TypeError):
            q.assert_("f1")


class TestRetract:
    def test_no_reason(self):
        assert _roundtrip(q.retract("f1")) == 'RETRACT "f1"'

    def test_with_reason(self):
        dsl = _roundtrip(q.retract("f1", reason="outdated"))
        assert dsl == 'RETRACT "f1" REASON "outdated"'


class TestMerge:
    def test_basic(self):
        assert _roundtrip(q.merge("old", "canonical")) == 'MERGE NODE "old" INTO "canonical"'


class TestPropagate:
    def test_basic(self):
        dsl = _roundtrip(q.propagate("m1", field="confidence", depth=3))
        assert dsl == 'PROPAGATE "m1" FIELD confidence DEPTH 3'


class TestBindDiscardContext:
    def test_bind(self):
        assert _roundtrip(q.bind_context("sess1")) == 'BIND CONTEXT "sess1"'

    def test_discard(self):
        assert _roundtrip(q.discard_context("sess1")) == 'DISCARD CONTEXT "sess1"'


class TestForget:
    def test_basic(self):
        assert _roundtrip(q.forget("m1")) == 'FORGET NODE "m1"'


class TestConnectNode:
    def test_basic(self):
        assert _roundtrip(q.connect_node("m1")) == 'CONNECT NODE "m1"'

    def test_with_threshold(self):
        dsl = _roundtrip(q.connect_node("m1", threshold=0.9))
        assert dsl == 'CONNECT NODE "m1" THRESHOLD 0.9'


class TestIngest:
    def test_file_only(self):
        assert _roundtrip(q.ingest("report.pdf")) == 'INGEST "report.pdf"'

    def test_as_kind(self):
        dsl = _roundtrip(q.ingest("report.pdf", as_id="doc:q3", kind="report"))
        assert 'INGEST "report.pdf"' in dsl
        assert 'AS "doc:q3"' in dsl
        assert 'KIND "report"' in dsl

    def test_using_parser(self):
        dsl = _roundtrip(q.ingest("x.pdf", using="pymupdf4llm"))
        assert "USING pymupdf4llm" in dsl

    def test_using_vision(self):
        dsl = _roundtrip(q.ingest("scan.pdf", using="vision", vision_model="smolvlm2-2.2b"))
        assert 'USING VISION "smolvlm2-2.2b"' in dsl

    def test_vision_model_without_using_raises(self):
        with pytest.raises(ValueError):
            q.ingest("x.pdf", vision_model="smolvlm2-2.2b", using="pymupdf4llm")

    def test_unknown_using_raises(self):
        with pytest.raises(ValueError):
            q.ingest("x.pdf", using="fantasy")


class TestBatch:
    def test_compose_with_or(self):
        b = q.create_node("n1", kind="m") | q.create_node("n2", kind="m")
        dsl = b.dsl()
        assert 'CREATE NODE "n1"' in dsl
        assert 'CREATE NODE "n2"' in dsl

    def test_begin_commit_wrap(self):
        b = q.begin() | q.create_node("n1", kind="m") | q.commit()
        dsl = b.dsl()
        lines = dsl.split("\n")
        assert lines[0] == "BEGIN"
        assert lines[-1] == "COMMIT"
        assert 'CREATE NODE "n1"' in dsl
        parse(dsl)

    def test_batch_shorthand(self):
        b = q.batch(
            q.create_node("n1", kind="m"),
            q.create_node("n2", kind="m"),
            q.create_edge("n1", "n2", kind="next"),
        )
        dsl = b.dsl()
        assert dsl.startswith("BEGIN\n")
        assert dsl.endswith("\nCOMMIT")
        parse(dsl)

    def test_batch_empty_raises(self):
        with pytest.raises(ValueError):
            q.batch()


class TestWriteEscape:
    def test_update_node_escape(self):
        out = q.update_node("m1", note='a"b').dsl()
        assert r'\"' in out

    def test_retract_reason_escape(self):
        out = q.retract("f1", reason='he said "no"').dsl()
        assert r'\"' in out
