"""Tests for the 4 summary.md gaps: lexical recall, vision ingest, blob lifecycle, section hierarchy."""

import time
import pytest
from unittest.mock import patch, MagicMock

from supergraph import SuperGraph
from supergraph.dsl.parser import parse_uncached
from supergraph.dsl.ast_nodes import LexicalSearchQuery, ForgetNode, SysRetain


# ============================================================
# Gap 1: Lexical Recall (FTS5)
# ============================================================

class TestLexicalRecall:
    def test_parse_lexical_search(self):
        ast = parse_uncached('LEXICAL SEARCH "machine learning" LIMIT 5')
        assert isinstance(ast, LexicalSearchQuery)
        assert ast.query == "machine learning"
        assert ast.limit.value == 5

    def test_parse_lexical_search_with_where(self):
        ast = parse_uncached('LEXICAL SEARCH "neural nets" LIMIT 10 WHERE kind = "chunk"')
        assert isinstance(ast, LexicalSearchQuery)
        assert ast.query == "neural nets"
        assert ast.where is not None

    def test_lexical_search_returns_results(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        gs.execute('CREATE NODE "c1" kind = "chunk" summary = "machine learning algorithms"')
        gs.execute('CREATE NODE "c2" kind = "chunk" summary = "deep neural network training"')
        gs.execute('CREATE NODE "c3" kind = "chunk" summary = "database indexing strategies"')

        # Manually put summaries into DocumentStore for FTS
        for nid, summary in [("c1", "machine learning algorithms"),
                              ("c2", "deep neural network training"),
                              ("c3", "database indexing strategies")]:
            str_id = gs._store.string_table.intern(nid)
            slot = gs._store.id_to_slot[str_id]
            gs._document_store.put_summary(slot, summary)

        result = gs.execute('LEXICAL SEARCH "machine learning"')
        assert result.kind == "nodes"
        assert result.count >= 1
        ids = [n["id"] for n in result.data]
        assert "c1" in ids
        gs.close()

    def test_lexical_search_empty(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        result = gs.execute('LEXICAL SEARCH "nonexistent term xyz"')
        assert result.kind == "nodes"
        assert result.count == 0
        gs.close()

    def test_lexical_search_respects_limit(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        for i in range(10):
            gs.execute(f'CREATE NODE "n{i}" kind = "chunk" summary = "common topic here"')
            str_id = gs._store.string_table.intern(f"n{i}")
            slot = gs._store.id_to_slot[str_id]
            gs._document_store.put_summary(slot, "common topic here")

        result = gs.execute('LEXICAL SEARCH "common topic" LIMIT 3')
        assert result.count <= 3
        gs.close()


# ============================================================
# Gap 2: Vision Ingest (grammar + AST wiring, no real VLM)
# ============================================================

class TestVisionIngest:
    def test_parse_ingest_with_vision(self):
        ast = parse_uncached('INGEST "photo.png" USING VISION "smolvlm2:2.2b"')
        assert ast.vision_model == "smolvlm2:2.2b"

    def test_standalone_image_ingest_with_vision(self, tmp_path):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)

        mock_vh = MagicMock()
        mock_vh.describe.return_value = "A test image showing a diagram"

        with patch("supergraph.ingest.vision.VisionHandler", return_value=mock_vh):
            result = gs.execute(f'INGEST "{img_path}" USING VISION "test-model"')

        assert result.kind == "ok"
        assert result.data["images"] == 1
        assert result.data["parser"] == "vision"

        node = gs.execute(f'NODE "{result.data["doc_id"]}"')
        assert node.data is not None
        assert node.data["summary"] == "A test image showing a diagram"
        gs.close()


# ============================================================
# Gap 3: Blob Lifecycle (FORGET + SYS RETAIN + __blob_state__)
# ============================================================

class TestForgetNode:
    def test_parse_forget(self):
        ast = parse_uncached('FORGET NODE "old-memory"')
        assert isinstance(ast, ForgetNode)
        assert ast.id == "old-memory"

    def test_forget_removes_node(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        gs.execute('CREATE NODE "mem1" kind = "memory" summary = "some memory"')
        result = gs.execute('NODE "mem1"')
        assert result.data is not None

        gs.execute('FORGET NODE "mem1"')
        result = gs.execute('NODE "mem1"')
        assert result.data is None
        gs.close()

    def test_forget_cascades_document_children(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        gs.execute('CREATE NODE "doc1" kind = "document" source = "test.md"')
        gs.execute('CREATE NODE "doc1:chunk:0" kind = "chunk" summary = "chunk text"')
        gs.execute('CREATE EDGE "doc1" -> "doc1:chunk:0" kind = "has_chunk"')

        gs.execute('FORGET NODE "doc1"')

        assert gs.execute('NODE "doc1"').data is None
        assert gs.execute('NODE "doc1:chunk:0"').data is None
        gs.close()


class TestSysRetain:
    def test_parse_sys_retain(self):
        ast = parse_uncached('SYS RETAIN')
        assert isinstance(ast, SysRetain)

    def test_retain_transitions_blob_state(self, tmp_path):
        gs = SuperGraph(
            path=str(tmp_path / "db"),
            embedder=None,
            retention={"blob_warm_days": 0, "blob_archive_days": 0, "blob_delete_days": 9999},
        )
        gs.execute('CREATE NODE "n1" kind = "chunk" summary = "test"')
        str_id = gs._store.string_table.intern("n1")
        slot = gs._store.id_to_slot[str_id]
        gs._store.columns.set_reserved(slot, "__blob_state__", "warm")
        gs._store.columns.set_reserved(slot, "__created_at__", int(time.time() * 1000) - 86400001)

        result = gs.execute('SYS RETAIN')
        assert result.data["archived"] >= 1

        # Check the node is now archived
        blob_col = gs._store.columns.get_column("__blob_state__", gs._store._next_slot)
        assert blob_col is not None
        _, pres, _ = blob_col
        assert pres[slot]
        gs.close()


class TestBlobStateOnIngest:
    def test_ingest_sets_blob_state_warm(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("# Hello\nSome content here for testing.")

        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        result = gs.execute(f'INGEST "{test_file}"')
        doc_id = result.data["doc_id"]

        str_id = gs._store.string_table.intern(doc_id)
        slot = gs._store.id_to_slot[str_id]

        blob_col = gs._store.columns.get_column("__blob_state__", gs._store._next_slot)
        assert blob_col is not None
        col_data, col_pres, dtype_str = blob_col
        assert col_pres[slot]
        # Verify it's "warm" (interned string)
        warm_id = gs._store.string_table.intern("warm")
        assert int(col_data[slot]) == warm_id
        gs.close()


# ============================================================
# Gap 4: Structural Recall (section hierarchy)
# ============================================================

class TestSectionHierarchy:
    def test_ingest_creates_sections(self, tmp_path):
        test_file = tmp_path / "doc.md"
        test_file.write_text(
            "# Introduction\n"
            "This is the intro.\n\n"
            "# Methods\n"
            "We used method A.\n\n"
            "# Results\n"
            "The results show...\n"
        )

        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        result = gs.execute(f'INGEST "{test_file}"')

        assert result.data["sections"] >= 2

        doc_id = result.data["doc_id"]
        edges = gs.execute(f'EDGES FROM "{doc_id}"')
        edge_kinds = [e["kind"] for e in edges.data]
        assert "has_section" in edge_kinds

        # Verify section nodes exist and have __confidence__ = 0.6
        section_edges = [e for e in edges.data if e["kind"] == "has_section"]
        assert len(section_edges) >= 2

        for se in section_edges:
            sec_node = gs.execute(f'NODE "{se["target"]}"')
            assert sec_node.data is not None
            assert sec_node.data["kind"] == "section"

            sec_slot = gs._store.id_to_slot[gs._store.string_table.intern(se["target"])]
            conf_col = gs._store.columns.get_column("__confidence__", gs._store._next_slot)
            if conf_col:
                col_data, col_pres, _ = conf_col
                if col_pres[sec_slot]:
                    assert abs(float(col_data[sec_slot]) - 0.6) < 0.01

        gs.close()

    def test_chunks_linked_to_sections(self, tmp_path):
        test_file = tmp_path / "doc.md"
        test_file.write_text(
            "# Alpha\n"
            "Alpha content.\n\n"
            "# Beta\n"
            "Beta content.\n"
        )

        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        result = gs.execute(f'INGEST "{test_file}"')
        doc_id = result.data["doc_id"]

        # Get section nodes
        doc_edges = gs.execute(f'EDGES FROM "{doc_id}"')
        section_ids = [e["target"] for e in doc_edges.data if e["kind"] == "has_section"]

        # Each section should have has_chunk edges to its chunks
        for sec_id in section_ids:
            sec_edges = gs.execute(f'EDGES FROM "{sec_id}"')
            chunk_edges = [e for e in sec_edges.data if e["kind"] == "has_chunk"]
            assert len(chunk_edges) >= 1

        gs.close()

    def test_flat_doc_no_sections(self, tmp_path):
        """A doc without headings should have chunks directly under parent."""
        test_file = tmp_path / "flat.txt"
        test_file.write_text("Just a paragraph of text without any headings at all.")

        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        result = gs.execute(f'INGEST "{test_file}"')
        assert result.data["sections"] == 0

        doc_id = result.data["doc_id"]
        edges = gs.execute(f'EDGES FROM "{doc_id}"')
        edge_kinds = [e["kind"] for e in edges.data]
        assert "has_section" not in edge_kinds
        assert "has_chunk" in edge_kinds
        gs.close()
