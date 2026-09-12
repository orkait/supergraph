"""Tests for SIMILAR TO query and vector integration."""
import numpy as np
import pytest
from supergraph import SuperGraph


class TestSimilarToByVector:
    def test_similar_by_vector(self):
        g = SuperGraph(embedder=None)
        g._ensure_vector_store(4)
        g.execute('CREATE NODE "m1" kind = "test" VECTOR [1.0, 0.0, 0.0, 0.0]')
        g.execute('CREATE NODE "m2" kind = "test" VECTOR [0.9, 0.1, 0.0, 0.0]')
        g.execute('CREATE NODE "m3" kind = "test" VECTOR [0.0, 0.0, 1.0, 0.0]')
        result = g.execute('SIMILAR TO [1.0, 0.0, 0.0, 0.0] LIMIT 2')
        ids = [n["id"] for n in result.data]
        assert "m1" in ids
        assert "m2" in ids
        assert "m3" not in ids

    def test_similar_has_score(self):
        g = SuperGraph(embedder=None)
        g._ensure_vector_store(4)
        g.execute('CREATE NODE "m1" kind = "test" VECTOR [1.0, 0.0, 0.0, 0.0]')
        result = g.execute('SIMILAR TO [1.0, 0.0, 0.0, 0.0] LIMIT 5')
        assert len(result.data) >= 1
        assert "_similarity_score" in result.data[0]

    def test_similar_to_node(self):
        g = SuperGraph(embedder=None)
        g._ensure_vector_store(4)
        g.execute('CREATE NODE "q1" kind = "query" VECTOR [1.0, 0.0, 0.0, 0.0]')
        g.execute('CREATE NODE "m1" kind = "test" VECTOR [0.9, 0.1, 0.0, 0.0]')
        result = g.execute('SIMILAR TO NODE "q1" LIMIT 5')
        assert len(result.data) >= 1

    def test_similar_respects_retracted(self):
        g = SuperGraph(embedder=None)
        g._ensure_vector_store(4)
        g.execute('CREATE NODE "m1" kind = "test" VECTOR [1.0, 0.0, 0.0, 0.0]')
        g.execute('RETRACT "m1"')
        result = g.execute('SIMILAR TO [1.0, 0.0, 0.0, 0.0] LIMIT 5')
        assert len(result.data) == 0

    def test_similar_with_where(self):
        g = SuperGraph(embedder=None)
        g._ensure_vector_store(4)
        g.execute('CREATE NODE "m1" kind = "fact" VECTOR [1.0, 0.0, 0.0, 0.0]')
        g.execute('CREATE NODE "m2" kind = "memory" VECTOR [0.9, 0.1, 0.0, 0.0]')
        result = g.execute('SIMILAR TO [1.0, 0.0, 0.0, 0.0] LIMIT 5 WHERE kind = "fact"')
        assert all(n["kind"] == "fact" for n in result.data)

    def test_similar_empty_index(self):
        g = SuperGraph(embedder=None)
        result = g.execute('SIMILAR TO [1.0, 0.0, 0.0, 0.0] LIMIT 5')
        assert len(result.data) == 0


@pytest.mark.needs_embedder
class TestSimilarToByText:
    def test_similar_by_text(self):
        g = SuperGraph()  # default model2vec embedder
        g.execute('SYS REGISTER NODE KIND "memory" REQUIRED content:string EMBED content')
        g.execute('CREATE NODE "m1" kind = "memory" content = "Eiffel Tower at sunset"')
        g.execute('CREATE NODE "m2" kind = "memory" content = "Louvre museum in Paris"')
        g.execute('CREATE NODE "m3" kind = "memory" content = "quantum physics equations"')
        result = g.execute('SIMILAR TO "Paris travel" LIMIT 2')
        ids = [n["id"] for n in result.data]
        # m3 (quantum physics) should not be in top-2 for "Paris travel"
        assert "m3" not in ids

    def test_auto_embed_on_create(self):
        g = SuperGraph()
        g.execute('SYS REGISTER NODE KIND "memory" REQUIRED content:string EMBED content')
        g.execute('CREATE NODE "m1" kind = "memory" content = "hello world"')
        # Vector should exist
        assert g._vector_store is not None
        assert g._vector_store.has_vector(0)


class TestVectorClause:
    def test_create_with_vector(self):
        g = SuperGraph(embedder=None)
        g.execute('CREATE NODE "m1" kind = "test" VECTOR [0.5, 0.5, 0.0, 0.0]')
        assert g._vector_store is not None
        assert g._vector_store.has_vector(0)

    def test_explicit_vector_overrides_auto_embed(self):
        """Explicit VECTOR clause should be used instead of auto-embedding."""
        g = SuperGraph(embedder=None)
        g._ensure_vector_store(4)
        g.execute('SYS REGISTER NODE KIND "memory" REQUIRED content:string')
        g.execute('CREATE NODE "m1" kind = "memory" content = "hello" VECTOR [1.0, 0.0, 0.0, 0.0]')
        # Should use explicit vector, not auto-embedded
        vec = g._vector_store.get_vector(0)
        assert len(vec) == 4
        assert np.allclose(vec, [1.0, 0.0, 0.0, 0.0])
        g.close()


class TestEmbedClause:
    def test_register_with_embed(self):
        g = SuperGraph()
        g.execute('SYS REGISTER NODE KIND "memory" REQUIRED content:string, topic:string EMBED content')
        desc = g.execute('SYS DESCRIBE NODE "memory"')
        assert desc.data is not None
        assert desc.data.get("embed_field") == "content"
