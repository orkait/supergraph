"""Tests for SYS CONNECT, SYS REEMBED, RETRACT/DELETE cascade, and SYS STATUS."""
import pytest
from supergraph import SuperGraph


class TestSysConnect:
    def test_connect_creates_edges(self, tmp_path):
        g = SuperGraph(path=str(tmp_path / "db"))
        g.execute('CREATE NODE "a" kind = "chunk" summary = "Paris travel" VECTOR [1.0, 0.0, 0.0, 0.0]')
        g.execute('CREATE NODE "b" kind = "chunk" summary = "France tourism" VECTOR [0.95, 0.05, 0.0, 0.0]')
        g.execute('CREATE NODE "c" kind = "chunk" summary = "quantum physics" VECTOR [0.0, 0.0, 1.0, 0.0]')
        result = g.execute('SYS CONNECT THRESHOLD 0.8')
        assert result.data["edges_created"] >= 1

    def test_connect_idempotent(self, tmp_path):
        g = SuperGraph(path=str(tmp_path / "db"))
        g.execute('CREATE NODE "a" kind = "chunk" VECTOR [1.0, 0.0, 0.0, 0.0]')
        g.execute('CREATE NODE "b" kind = "chunk" VECTOR [0.95, 0.05, 0.0, 0.0]')
        r1 = g.execute('SYS CONNECT THRESHOLD 0.8')
        r2 = g.execute('SYS CONNECT THRESHOLD 0.8')
        assert r2.data["edges_created"] == 0  # already connected

    def test_connect_node(self, tmp_path):
        g = SuperGraph(path=str(tmp_path / "db"))
        g.execute('CREATE NODE "a" kind = "chunk" VECTOR [1.0, 0.0, 0.0, 0.0]')
        g.execute('CREATE NODE "b" kind = "chunk" VECTOR [0.9, 0.1, 0.0, 0.0]')
        result = g.execute('CONNECT NODE "a" THRESHOLD 0.8')
        assert result.data["edges_created"] >= 1

    def test_connect_no_vectors(self, tmp_path):
        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        g.execute('CREATE NODE "a" kind = "chunk"')
        g.execute('CREATE NODE "b" kind = "chunk"')
        result = g.execute('SYS CONNECT')
        assert result.data["edges_created"] == 0

    def test_connect_default_threshold(self, tmp_path):
        g = SuperGraph(path=str(tmp_path / "db"))
        g.execute('CREATE NODE "x" kind = "chunk" VECTOR [1.0, 0.0, 0.0, 0.0]')
        g.execute('CREATE NODE "y" kind = "chunk" VECTOR [0.99, 0.01, 0.0, 0.0]')
        result = g.execute('SYS CONNECT')
        # Default threshold is 0.85, similarity between x and y should be high
        assert result.data["edges_created"] >= 1


class TestRetractCascade:
    def test_retract_doc_retracts_chunks(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("# Part 1\nContent\n\n# Part 2\nMore")
        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        g.execute(f'INGEST "{f}" AS "doc:test"')
        chunks_before = g.execute('NODES WHERE kind = "chunk"')
        assert len(chunks_before.data) >= 2
        g.execute('RETRACT "doc:test"')
        chunks_after = g.execute('NODES WHERE kind = "chunk"')
        assert len(chunks_after.data) == 0  # all retracted

    def test_retract_non_doc_no_cascade(self, tmp_path):
        """Retracting a non-document node should not cascade."""
        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        g.execute('CREATE NODE "a" kind = "chunk" summary = "test"')
        g.execute('CREATE NODE "b" kind = "chunk" summary = "other"')
        g.execute('CREATE EDGE "a" -> "b" kind = "has_chunk"')
        g.execute('RETRACT "a"')
        # Node b should still be visible
        b_result = g.execute('NODE "b"')
        assert b_result.data is not None

    def test_delete_doc_cascades(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("# Part 1\nContent")
        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        g.execute(f'INGEST "{f}" AS "doc:del"')
        g.execute('DELETE NODE "doc:del"')
        assert len(g.execute('NODES WHERE kind = "chunk"').data) == 0
        assert g.execute('NODE "doc:del"').data is None

    def test_delete_non_doc_no_cascade(self, tmp_path):
        """Deleting a non-document node should not cascade."""
        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        g.execute('CREATE NODE "a" kind = "chunk" summary = "test"')
        g.execute('CREATE NODE "b" kind = "chunk" summary = "other"')
        g.execute('CREATE EDGE "a" -> "b" kind = "has_chunk"')
        g.execute('DELETE NODE "a"')
        # Node b should still exist
        b_result = g.execute('NODE "b"')
        assert b_result.data is not None


class TestSysReembed:
    def test_reembed_no_embedder_raises(self, tmp_path):
        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        with pytest.raises(Exception, match="No embedder"):
            g.execute('SYS REEMBED')

    def test_embedder_dirty_flag(self, tmp_path):
        """Dirty flag should block SIMILAR TO queries."""
        g = SuperGraph(path=str(tmp_path / "db"))
        g.execute('CREATE NODE "a" kind = "chunk" summary = "test" VECTOR [1.0, 0.0, 0.0, 0.0]')
        # Manually set dirty flag
        g._embedder_dirty = True
        with pytest.raises(Exception, match="Embedder changed"):
            g.execute('SIMILAR TO [1.0, 0.0, 0.0, 0.0]')

    @pytest.mark.needs_embedder
    def test_reembed_clears_dirty_flag(self, tmp_path):
        g = SuperGraph(path=str(tmp_path / "db"))
        # Create node using auto-embed (embedder creates proper vector dimensions)
        g.execute('CREATE NODE "a" kind = "chunk" summary = "hello world"')
        # Manually embed to initialize vector store with correct dims
        if g._embedder:
            import numpy as np
            vec = g._embedder.encode_documents(["hello world"])[0]
            g._ensure_vector_store(len(vec))
            slot = g._executor._resolve_slot("a")
            g._vector_store.add(slot, vec)
        g._embedder_dirty = True
        g.execute('SYS REEMBED')
        assert g._embedder_dirty is False


class TestSysStatus:
    def test_returns_all_fields(self, tmp_path):
        g = SuperGraph(path=str(tmp_path / "db"))
        result = g.execute('SYS STATUS')
        assert "nodes" in result.data
        assert "edges" in result.data
        assert "vectors" in result.data
        assert "embedder" in result.data

    def test_status_with_data(self, tmp_path):
        g = SuperGraph(path=str(tmp_path / "db"))
        g.execute('CREATE NODE "a" kind = "test" summary = "hello"')
        g.execute('CREATE NODE "b" kind = "test" summary = "world"')
        g.execute('CREATE EDGE "a" -> "b" kind = "related"')
        result = g.execute('SYS STATUS')
        assert result.data["nodes"] == 2
        assert result.data["edges"] == 1

    def test_status_documents(self, tmp_path):
        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        result = g.execute('SYS STATUS')
        assert "documents" in result.data

    def test_status_edge_types(self, tmp_path):
        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        g.execute('CREATE NODE "a" kind = "test"')
        g.execute('CREATE NODE "b" kind = "test"')
        g.execute('CREATE EDGE "a" -> "b" kind = "knows"')
        result = g.execute('SYS STATUS')
        assert "edge_types" in result.data
        assert "knows" in result.data["edge_types"]

    def test_status_uptime(self, tmp_path):
        g = SuperGraph(path=str(tmp_path / "db"))
        result = g.execute('SYS STATUS')
        assert result.data["uptime_seconds"] >= 0
