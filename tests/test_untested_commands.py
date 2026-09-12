import tempfile
import pytest
from supergraph import SuperGraph


class TestWeightedShortestPath:

    def test_basic_weighted_path(self):
        gs = SuperGraph()
        gs.execute('CREATE NODE "a" kind = "node"')
        gs.execute('CREATE NODE "b" kind = "node"')
        gs.execute('CREATE NODE "c" kind = "node"')
        gs.execute('CREATE EDGE "a" -> "b" kind = "road" weight = 10')
        gs.execute('CREATE EDGE "a" -> "c" kind = "road" weight = 1')
        gs.execute('CREATE EDGE "c" -> "b" kind = "road" weight = 1')

        result = gs.execute('WEIGHTED SHORTEST PATH FROM "a" TO "b"')
        assert result.kind == "path"
        assert result.data is not None
        assert result.data == ["a", "c", "b"]
        gs.close()

    def test_weighted_path_direct_is_cheapest(self):
        gs = SuperGraph()
        gs.execute('CREATE NODE "x" kind = "node"')
        gs.execute('CREATE NODE "y" kind = "node"')
        gs.execute('CREATE NODE "z" kind = "node"')
        gs.execute('CREATE EDGE "x" -> "y" kind = "road" weight = 1')
        gs.execute('CREATE EDGE "x" -> "z" kind = "road" weight = 100')
        gs.execute('CREATE EDGE "z" -> "y" kind = "road" weight = 100')

        result = gs.execute('WEIGHTED SHORTEST PATH FROM "x" TO "y"')
        assert result.data == ["x", "y"]
        gs.close()

    def test_weighted_path_no_path(self):
        gs = SuperGraph()
        gs.execute('CREATE NODE "isolated1" kind = "node"')
        gs.execute('CREATE NODE "isolated2" kind = "node"')
        result = gs.execute('WEIGHTED SHORTEST PATH FROM "isolated1" TO "isolated2"')
        assert result.data is None
        gs.close()

    def test_weighted_path_same_node(self):
        gs = SuperGraph()
        gs.execute('CREATE NODE "self" kind = "node"')
        gs.execute('CREATE NODE "other" kind = "node"')
        gs.execute('CREATE EDGE "self" -> "other" kind = "link" weight = 1')
        result = gs.execute('WEIGHTED SHORTEST PATH FROM "self" TO "self"')
        assert result.data is not None
        assert result.data == ["self"]
        gs.close()

    def test_weighted_path_nonexistent_node(self):
        gs = SuperGraph()
        result = gs.execute('WEIGHTED SHORTEST PATH FROM "ghost" TO "phantom"')
        assert result.data is None
        gs.close()


class TestWeightedDistance:

    def test_basic_weighted_distance(self):
        gs = SuperGraph()
        gs.execute('CREATE NODE "a" kind = "node"')
        gs.execute('CREATE NODE "b" kind = "node"')
        gs.execute('CREATE NODE "c" kind = "node"')
        gs.execute('CREATE EDGE "a" -> "c" kind = "road" weight = 3')
        gs.execute('CREATE EDGE "c" -> "b" kind = "road" weight = 4')

        result = gs.execute('WEIGHTED DISTANCE FROM "a" TO "b"')
        assert result.kind == "distance"
        assert result.data == 7.0
        gs.close()

    def test_weighted_distance_no_path(self):
        gs = SuperGraph()
        gs.execute('CREATE NODE "x" kind = "node"')
        gs.execute('CREATE NODE "y" kind = "node"')
        result = gs.execute('WEIGHTED DISTANCE FROM "x" TO "y"')
        assert result.data == -1
        gs.close()

    def test_weighted_distance_same_node(self):
        gs = SuperGraph()
        gs.execute('CREATE NODE "s" kind = "node"')
        result = gs.execute('WEIGHTED DISTANCE FROM "s" TO "s"')
        assert result.data == 0.0
        gs.close()

    def test_weighted_distance_default_weight(self):
        gs = SuperGraph()
        gs.execute('CREATE NODE "p" kind = "node"')
        gs.execute('CREATE NODE "q" kind = "node"')
        gs.execute('CREATE NODE "r" kind = "node"')
        gs.execute('CREATE EDGE "p" -> "q" kind = "link"')
        gs.execute('CREATE EDGE "q" -> "r" kind = "link"')

        result = gs.execute('WEIGHTED DISTANCE FROM "p" TO "r"')
        assert result.data == 2.0
        gs.close()


class TestUpdateEdge:

    def test_update_edge_fields(self):
        gs = SuperGraph()
        gs.execute('CREATE NODE "a" kind = "node"')
        gs.execute('CREATE NODE "b" kind = "node"')
        gs.execute('CREATE EDGE "a" -> "b" kind = "friend" weight = 5')

        gs.execute('UPDATE EDGE "a" -> "b" SET weight = 99 WHERE kind = "friend"')

        edges = gs.execute('EDGES FROM "a" WHERE kind = "friend"')
        assert len(edges.data) == 1
        assert edges.data[0]["weight"] == 99
        gs.close()

    def test_update_edge_add_field(self):
        gs = SuperGraph()
        gs.execute('CREATE NODE "x" kind = "node"')
        gs.execute('CREATE NODE "y" kind = "node"')
        gs.execute('CREATE EDGE "x" -> "y" kind = "link"')

        gs.execute('UPDATE EDGE "x" -> "y" SET label = "new" WHERE kind = "link"')

        edges = gs.execute('EDGES FROM "x" WHERE kind = "link"')
        assert len(edges.data) == 1
        assert edges.data[0]["label"] == "new"
        gs.close()

    def test_update_edge_no_match(self):
        gs = SuperGraph()
        gs.execute('CREATE NODE "a" kind = "node"')
        gs.execute('CREATE NODE "b" kind = "node"')
        gs.execute('CREATE EDGE "a" -> "b" kind = "friend"')

        result = gs.execute('UPDATE EDGE "a" -> "b" SET x = 1 WHERE kind = "enemy"')
        assert result.data["updated"] == 0
        gs.close()

    def test_update_edge_without_where(self):
        gs = SuperGraph()
        gs.execute('CREATE NODE "a" kind = "node"')
        gs.execute('CREATE NODE "b" kind = "node"')
        gs.execute('CREATE EDGE "a" -> "b" kind = "type1"')
        gs.execute('CREATE EDGE "a" -> "b" kind = "type2" weight = 5')

        result = gs.execute('UPDATE EDGE "a" -> "b" SET tag = "updated"')
        assert result.data["updated"] >= 1
        gs.close()


class TestForgetNode:

    def test_forget_removes_node(self):
        gs = SuperGraph()
        gs.execute('CREATE NODE "ephemeral" kind = "temp" data = "secret"')
        result = gs.execute('NODE "ephemeral"')
        assert result.data is not None

        gs.execute('FORGET NODE "ephemeral"')
        result = gs.execute('NODE "ephemeral"')
        assert result.data is None
        gs.close()

    def test_forget_removes_edges(self):
        gs = SuperGraph()
        gs.execute('CREATE NODE "a" kind = "node"')
        gs.execute('CREATE NODE "b" kind = "node"')
        gs.execute('CREATE EDGE "a" -> "b" kind = "link"')

        gs.execute('FORGET NODE "a"')
        edges = gs.execute('EDGES TO "b"')
        assert len(edges.data) == 0
        gs.close()

    def test_forget_cascades_document(self):
        with tempfile.TemporaryDirectory() as td:
            gs = SuperGraph(path=td)
            gs.execute('CREATE NODE "doc:test" kind = "document" source = "test.txt"')
            gs.execute('CREATE NODE "doc:test:chunk:0" kind = "chunk" summary = "chunk zero"')
            gs.execute('CREATE EDGE "doc:test" -> "doc:test:chunk:0" kind = "has_chunk"')

            gs.execute('FORGET NODE "doc:test"')

            result = gs.execute('NODE "doc:test"')
            assert result.data is None
            result = gs.execute('NODE "doc:test:chunk:0"')
            assert result.data is None
            gs.close()

    def test_forget_nonexistent_raises(self):
        gs = SuperGraph()
        with pytest.raises(Exception):
            gs.execute('FORGET NODE "ghost"')
        gs.close()

    def test_forget_with_vector(self):
        from supergraph.embedding.base import Embedder
        import numpy as np

        class TinyEmbedder(Embedder):
            @property
            def name(self): return "tiny"
            @property
            def dims(self): return 4
            def encode_documents(self, texts, titles=None):
                return np.random.randn(len(texts), 4).astype(np.float32)
            def encode_queries(self, texts):
                return np.random.randn(len(texts), 4).astype(np.float32)

        gs = SuperGraph(embedder=TinyEmbedder())
        gs.execute('SYS REGISTER NODE KIND "item" REQUIRED text:string EMBED text')
        gs.execute('CREATE NODE "vec_node" kind = "item" text = "hello world"')

        assert gs._vector_store is not None
        slot = gs._store.id_to_slot[gs._store.string_table.intern("vec_node")]
        assert gs._vector_store.has_vector(slot)

        gs.execute('FORGET NODE "vec_node"')

        assert not gs._vector_store.has_vector(slot)
        gs.close()
