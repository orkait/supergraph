import pytest
from supergraph import SuperGraph


class TestSysDuplicates:
    def test_finds_near_duplicates(self):
        g = SuperGraph(embedder=None)
        g._ensure_vector_store(4)
        g.execute('CREATE NODE "a" kind = "test" VECTOR [1.0, 0.0, 0.0, 0.0]')
        g.execute('CREATE NODE "b" kind = "test" VECTOR [0.99, 0.01, 0.0, 0.0]')
        g.execute('CREATE NODE "c" kind = "test" VECTOR [0.0, 0.0, 1.0, 0.0]')
        result = g.execute('SYS DUPLICATES THRESHOLD 0.9')
        assert result.count >= 1
        pair = result.data[0]
        assert "node_a" in pair and "node_b" in pair
        assert pair["similarity"] > 0.9

    def test_no_duplicates(self):
        g = SuperGraph(embedder=None)
        g._ensure_vector_store(4)
        g.execute('CREATE NODE "a" kind = "test" VECTOR [1.0, 0.0, 0.0, 0.0]')
        g.execute('CREATE NODE "b" kind = "test" VECTOR [0.0, 1.0, 0.0, 0.0]')
        result = g.execute('SYS DUPLICATES THRESHOLD 0.99')
        assert result.count == 0

    def test_duplicates_with_where(self):
        g = SuperGraph(embedder=None)
        g._ensure_vector_store(4)
        g.execute('CREATE NODE "a" kind = "memory" VECTOR [1.0, 0.0, 0.0, 0.0]')
        g.execute('CREATE NODE "b" kind = "memory" VECTOR [0.99, 0.01, 0.0, 0.0]')
        g.execute('CREATE NODE "c" kind = "fact" VECTOR [1.0, 0.0, 0.0, 0.0]')
        result = g.execute('SYS DUPLICATES WHERE kind = "memory" THRESHOLD 0.9')
        for pair in result.data:
            assert pair["node_a"] != "c" and pair["node_b"] != "c"

    def test_duplicates_default_threshold(self):
        g = SuperGraph(embedder=None)
        g._ensure_vector_store(4)
        g.execute('CREATE NODE "a" kind = "test" VECTOR [1.0, 0.0, 0.0, 0.0]')
        g.execute('CREATE NODE "b" kind = "test" VECTOR [0.99, 0.01, 0.0, 0.0]')
        result = g.execute('SYS DUPLICATES')
        assert result.count >= 1

    def test_deduplicates_pairs(self):
        g = SuperGraph(embedder=None)
        g._ensure_vector_store(4)
        g.execute('CREATE NODE "a" kind = "test" VECTOR [1.0, 0.0, 0.0, 0.0]')
        g.execute('CREATE NODE "b" kind = "test" VECTOR [0.99, 0.01, 0.0, 0.0]')
        result = g.execute('SYS DUPLICATES THRESHOLD 0.9')
        pairs = [(p["node_a"], p["node_b"]) for p in result.data]
        reverse_pairs = [(b, a) for a, b in pairs]
        for rp in reverse_pairs:
            assert rp not in pairs


class TestSysEmbedders:
    @pytest.mark.needs_embedder
    def test_lists_active_embedder(self):
        g = SuperGraph()
        result = g.execute('SYS EMBEDDERS')
        assert result.count >= 1
        assert result.data[0]["name"] == "model2vec"

    def test_no_embedder(self):
        g = SuperGraph(embedder=None)
        result = g.execute('SYS EMBEDDERS')
        assert result.data[0]["status"] == "no embedder configured"


class TestVectorPersistence:
    def test_roundtrip(self, tmp_path):
        g = SuperGraph(path=str(tmp_path), embedder=None)
        g._ensure_vector_store(4)
        g.execute('CREATE NODE "m1" kind = "test" VECTOR [1.0, 0.0, 0.0, 0.0]')
        g.execute('CREATE NODE "m2" kind = "test" VECTOR [0.0, 1.0, 0.0, 0.0]')
        g.checkpoint()
        g.close()

        g2 = SuperGraph(path=str(tmp_path), embedder=None)
        result = g2.execute('SIMILAR TO [1.0, 0.0, 0.0, 0.0] LIMIT 5')
        assert len(result.data) >= 1
        assert result.data[0]["id"] == "m1"
        g2.close()

    def test_snapshot_rollback_with_vectors(self):
        g = SuperGraph(embedder=None)
        g._ensure_vector_store(4)
        g.execute('CREATE NODE "m1" kind = "test" VECTOR [1.0, 0.0, 0.0, 0.0]')
        g.execute('SYS SNAPSHOT "before"')
        g.execute('CREATE NODE "m2" kind = "test" VECTOR [0.0, 1.0, 0.0, 0.0]')
        g.execute('SYS ROLLBACK TO "before"')
        result = g.execute('SIMILAR TO [0.0, 1.0, 0.0, 0.0] LIMIT 5')
        ids = [n["id"] for n in result.data]
        assert "m2" not in ids
