import time
import pytest
from supergraph import SuperGraph
from supergraph.core.errors import NodeNotFound


class TestAssert:
    def test_assert_creates_node(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('ASSERT "fact:1" kind = "fact" value = 42 CONFIDENCE 0.9 SOURCE "test"')
        node = g.execute('NODE "fact:1"')
        assert node.data is not None
        assert node.data["value"] == 42

    def test_assert_upserts_existing(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('ASSERT "f1" kind = "fact" value = 1 CONFIDENCE 0.5')
        g.execute('ASSERT "f1" kind = "fact" value = 2 CONFIDENCE 0.9')
        node = g.execute('NODE "f1"')
        assert node.data["value"] == 2

    def test_assert_without_confidence(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('ASSERT "f1" kind = "fact" value = 1')
        node = g.execute('NODE "f1"')
        assert node.data is not None


class TestRetract:
    def test_retract_hides_node(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "f1" kind = "fact" value = "old"')
        g.execute('RETRACT "f1" REASON "outdated"')
        assert g.execute('NODE "f1"').data is None
        assert len(g.execute('NODES').data) == 0

    def test_retract_nonexistent_raises(self):
        g = SuperGraph(ceiling_mb=256)
        with pytest.raises(NodeNotFound):
            g.execute('RETRACT "nonexistent"')

    def test_retract_without_reason(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "f1" kind = "fact" value = "x"')
        g.execute('RETRACT "f1"')
        assert g.execute('NODE "f1"').data is None


class TestUpdateNodesWhere:
    def test_bulk_update(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('SYS REGISTER NODE KIND "fact" REQUIRED confidence:float')
        for i in range(10):
            g.execute(f'CREATE NODE "f{i}" kind = "fact" confidence = 0.9')
        result = g.execute('UPDATE NODES WHERE kind = "fact" SET confidence = 0.1')
        assert result.data["updated"] == 10
        for i in range(10):
            node = g.execute(f'NODE "f{i}"')
            assert abs(node.data["confidence"] - 0.1) < 1e-10

    def test_update_nodes_partial_match(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('SYS REGISTER NODE KIND "a" REQUIRED score:int')
        g.execute('SYS REGISTER NODE KIND "b" REQUIRED score:int')
        g.execute('CREATE NODE "a1" kind = "a" score = 1')
        g.execute('CREATE NODE "b1" kind = "b" score = 2')
        result = g.execute('UPDATE NODES WHERE kind = "a" SET score = 99')
        assert result.data["updated"] == 1
        assert g.execute('NODE "a1"').data["score"] == 99
        assert g.execute('NODE "b1"').data["score"] == 2


class TestTTL:
    def test_expires_in_makes_invisible(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "tmp" kind = "working" name = "scratch" EXPIRES IN 0s')
        time.sleep(0.01)
        assert len(g.execute('NODES').data) == 0

    def test_expires_in_future_stays_visible(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "tmp" kind = "working" name = "scratch" EXPIRES IN 3600s')
        assert len(g.execute('NODES').data) == 1

    def test_sys_expire_tombstones(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "tmp" kind = "working" name = "scratch" EXPIRES IN 0s')
        time.sleep(0.01)
        result = g.execute('SYS EXPIRE')
        assert result.data["expired"] >= 1


class TestContradictions:
    def test_detects_conflicting_values(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('SYS REGISTER NODE KIND "belief" REQUIRED topic:string, value:string')
        g.execute('CREATE NODE "b1" kind = "belief" topic = "capital" value = "Paris"')
        g.execute('CREATE NODE "b2" kind = "belief" topic = "capital" value = "London"')
        g.execute('CREATE NODE "b3" kind = "belief" topic = "color" value = "blue"')
        result = g.execute('SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic')
        assert result.count == 1
        assert any(c["group"] == "capital" for c in result.data)

    def test_no_contradictions_when_consistent(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('SYS REGISTER NODE KIND "belief" REQUIRED topic:string, value:string')
        g.execute('CREATE NODE "b1" kind = "belief" topic = "x" value = "same"')
        g.execute('CREATE NODE "b2" kind = "belief" topic = "x" value = "same"')
        result = g.execute('SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic')
        assert result.count == 0


class TestMerge:
    def test_merge_copies_fields(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "src" kind = "fact" name = "old" extra = "data"')
        g.execute('CREATE NODE "tgt" kind = "fact" name = "canonical"')
        result = g.execute('MERGE NODE "src" INTO "tgt"')
        assert result.data["fields_merged"] >= 1
        tgt = g.execute('NODE "tgt"')
        assert tgt.data["name"] == "canonical"
        assert tgt.data["extra"] == "data"
        assert g.execute('NODE "src"').data is None

    def test_merge_rewires_edges(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "src" kind = "x"')
        g.execute('CREATE NODE "tgt" kind = "x"')
        g.execute('CREATE NODE "other" kind = "x"')
        g.execute('CREATE EDGE "src" -> "other" kind = "link"')
        g.execute('MERGE NODE "src" INTO "tgt"')
        edges = g.execute('EDGES FROM "tgt"')
        assert len(edges.data) >= 1
        assert any(e["target"] == "other" for e in edges.data)

    def test_merge_nonexistent_source_raises(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "tgt" kind = "x"')
        with pytest.raises(NodeNotFound):
            g.execute('MERGE NODE "nonexistent" INTO "tgt"')
