import os
import pytest
from supergraph.document.store import DocumentStore


@pytest.fixture
def doc_store(tmp_path):
    ds = DocumentStore(str(tmp_path / "test_docs.db"))
    yield ds
    ds.close()


@pytest.fixture
def temp_doc_store():
    ds = DocumentStore()
    yield ds
    ds.close()


class TestDocuments:
    def test_put_and_get(self, doc_store):
        doc_store.put_document(0, b"Hello world", "text/plain")
        content, ctype = doc_store.get_document(0)
        assert content == b"Hello world"
        assert ctype == "text/plain"

    def test_get_missing(self, doc_store):
        assert doc_store.get_document(999) is None

    def test_delete(self, doc_store):
        doc_store.put_document(0, b"data", "text/plain")
        doc_store.delete_document(0)
        assert doc_store.get_document(0) is None

    def test_has(self, doc_store):
        assert not doc_store.has_document(0)
        doc_store.put_document(0, b"data", "text/plain")
        assert doc_store.has_document(0)

    def test_replace(self, doc_store):
        doc_store.put_document(0, b"old", "text/plain")
        doc_store.put_document(0, b"new", "text/markdown")
        content, ctype = doc_store.get_document(0)
        assert content == b"new"

    def test_binary(self, doc_store):
        blob = bytes(range(256))
        doc_store.put_document(0, blob, "image/png")
        content, _ = doc_store.get_document(0)
        assert content == blob


class TestSummaries:
    def test_put_and_get(self, doc_store):
        doc_store.put_summary(10, "Q3 revenue grew 15%", heading="Revenue", page=7, chunk_index=3, doc_slot=0)
        s = doc_store.get_summary(10)
        assert s["summary"] == "Q3 revenue grew 15%"
        assert s["heading"] == "Revenue"
        assert s["page"] == 7

    def test_get_summaries_for_doc(self, doc_store):
        doc_store.put_summary(10, "chunk 1", doc_slot=0, chunk_index=0)
        doc_store.put_summary(11, "chunk 2", doc_slot=0, chunk_index=1)
        doc_store.put_summary(20, "other doc", doc_slot=1, chunk_index=0)
        summaries = doc_store.get_summaries_for_doc(0)
        assert len(summaries) == 2


class TestMetadata:
    def test_put_and_get(self, doc_store):
        doc_store.put_metadata(0, {"source_path": "/tmp/report.pdf", "pages": 42, "parser_used": "pymupdf4llm"})
        m = doc_store.get_metadata(0)
        assert m["source_path"] == "/tmp/report.pdf"
        assert m["pages"] == 42

    def test_missing(self, doc_store):
        assert doc_store.get_metadata(999) is None


class TestImages:
    def test_put_and_get(self, doc_store):
        doc_store.put_image(0, b"\x89PNG...", "image/png", page=3, description="A chart")
        img = doc_store.get_image(0)
        assert img["mime_type"] == "image/png"
        assert img["description"] == "A chart"
        assert img["page"] == 3


class TestBulkOps:
    def test_delete_all_for_doc(self, doc_store):
        doc_store.put_document(0, b"parent", "text/markdown")
        doc_store.put_summary(10, "chunk 1", doc_slot=0)
        doc_store.put_summary(11, "chunk 2", doc_slot=0)
        doc_store.put_document(10, b"chunk text 1", "text/markdown")
        doc_store.put_document(11, b"chunk text 2", "text/markdown")
        doc_store.put_metadata(0, {"pages": 5})
        count = doc_store.delete_all_for_doc(0)
        assert count >= 4
        assert doc_store.get_document(0) is None
        assert doc_store.get_summary(10) is None
        assert doc_store.get_metadata(0) is None

    def test_orphan_cleanup(self, doc_store):
        doc_store.put_document(0, b"live", "text/plain")
        doc_store.put_document(1, b"orphan", "text/plain")
        doc_store.put_summary(2, "orphan summary", doc_slot=1)
        cleaned = doc_store.orphan_cleanup(live_slots={0})
        assert cleaned >= 2
        assert doc_store.get_document(0) is not None
        assert doc_store.get_document(1) is None

    def test_stats(self, doc_store):
        doc_store.put_document(0, b"hello", "text/plain")
        doc_store.put_document(1, b"world!", "text/plain")
        doc_store.put_summary(10, "summary", doc_slot=0)
        doc_store.put_image(20, b"img", "image/png")
        s = doc_store.stats()
        assert s["document_count"] == 2
        assert s["total_bytes"] == 11
        assert s["chunk_count"] == 1
        assert s["image_count"] == 1


class TestSearchText:

    def _seed(self, doc_store):
        doc_store.put_summary(1, "I live in Tokyo and work as a data scientist", doc_slot=0)
        doc_store.put_summary(2, "My favorite food is tonkotsu ramen", doc_slot=0)
        doc_store.put_summary(3, "I adopted a cat named Miso", doc_slot=0)
        doc_store.put_summary(4, "Planning a trip to Hokkaido in December", doc_slot=0)

    def test_question_with_question_mark(self, doc_store):
        self._seed(doc_store)
        hits = doc_store.search_text("Where does the user live?", limit=5)
        assert 1 in {s for s, _ in hits}

    def test_question_with_apostrophe(self, doc_store):
        self._seed(doc_store)
        hits = doc_store.search_text("What is the user's favorite food?", limit=5)
        assert 2 in {s for s, _ in hits}

    def test_query_with_double_quotes(self, doc_store):
        self._seed(doc_store)
        hits = doc_store.search_text('Did the user say "Tokyo"?', limit=5)
        assert 1 in {s for s, _ in hits}

    def test_query_with_fts5_operators(self, doc_store):
        self._seed(doc_store)
        hits = doc_store.search_text("cat* (Miso) ^ +adopted -dog ~test", limit=5)
        assert 3 in {s for s, _ in hits}

    def test_empty_query_returns_empty(self, doc_store):
        self._seed(doc_store)
        assert doc_store.search_text("", limit=5) == []
        assert doc_store.search_text("   ", limit=5) == []

    def test_operator_only_query_returns_empty(self, doc_store):
        self._seed(doc_store)
        assert doc_store.search_text("?*^()", limit=5) == []

    def test_natural_question_returns_hits(self, doc_store):
        self._seed(doc_store)
        hits = doc_store.search_text("Where is the user planning to travel?", limit=5)
        assert hits, "expected at least one hit for a natural-language question"
        slots = {s for s, _ in hits}
        assert 4 in slots


class TestTempFile:
    def test_temp_mode_works(self, temp_doc_store):
        temp_doc_store.put_document(0, b"temp data", "text/plain")
        assert temp_doc_store.get_document(0) is not None

    def test_temp_file_cleaned_on_close(self):
        ds = DocumentStore()
        path = ds._path
        ds.put_document(0, b"data", "text/plain")
        assert os.path.exists(path)
        ds.close()
        assert not os.path.exists(path)
