"""Tests for extensibility injection points."""
import pytest
from supergraph.ingest.base import Ingestor, IngestResult


class _DummyIngestor(Ingestor):
    name = "dummy"
    supported_extensions = ["xyz"]

    def convert(self, file_path: str, **kwargs) -> IngestResult:
        return IngestResult(markdown="dummy", parser_used="dummy")


class TestIngestorRegistry:
    def test_register_and_resolve_by_extension(self):
        from supergraph.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        ingestor = _DummyIngestor()
        reg.register(ingestor)
        resolved = reg.resolve("file.xyz")
        assert resolved is ingestor

    def test_resolve_unknown_extension_raises(self):
        from supergraph.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        with pytest.raises(ValueError, match="Unsupported format"):
            reg.resolve("file.unknownext999")

    def test_resolve_using_unknown_name_raises(self):
        from supergraph.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        # using= with an unknown builtin name raises ValueError from _make_builtin_ingestor
        with pytest.raises(ValueError):
            reg.resolve("file.txt", using="nonexistent_parser_xyz")

    def test_override_existing_extension(self):
        from supergraph.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()

        class MyPDFIngestor(Ingestor):
            name = "mypdf"
            supported_extensions = ["pdf"]
            def convert(self, path, **kwargs):
                return IngestResult(markdown="mypdf", parser_used="mypdf")

        reg.register(MyPDFIngestor())
        resolved = reg.resolve("report.pdf")
        assert resolved.name == "mypdf"

    def test_builtin_pdf_resolves(self):
        from supergraph.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        ingestor = reg.resolve("report.pdf")
        assert ingestor is not None
        assert ingestor.name in ("pymupdf4llm", "markitdown")

    @pytest.mark.needs_ingest
    def test_builtin_txt_resolves(self):
        from supergraph.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        ingestor = reg.resolve("notes.txt")
        assert ingestor.name == "markitdown"

    def test_list_returns_registered(self):
        from supergraph.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        reg.register(_DummyIngestor())
        entries = reg.list()
        names = [e["name"] for e in entries]
        assert "dummy" in names

    def test_using_override_bypasses_extension_map(self):
        from supergraph.ingest.registry import IngestorRegistry
        reg = IngestorRegistry()
        reg.register(_DummyIngestor())
        resolved = reg.resolve("report.pdf", using="dummy")
        assert resolved.name == "dummy"


class TestChunkerProtocol:
    def test_heading_chunker_implements_protocol(self):
        from supergraph.ingest.base import ChunkerProtocol
        from supergraph.ingest.chunker import HeadingChunker
        chunker = HeadingChunker()
        assert isinstance(chunker, ChunkerProtocol)

    def test_heading_chunker_chunks_text(self):
        from supergraph.ingest.chunker import HeadingChunker
        chunker = HeadingChunker()
        chunks = chunker.chunk("# Heading\nsome text")
        assert len(chunks) >= 1
        assert chunks[0].heading == "Heading"

    def test_heading_chunker_passes_kwargs(self):
        from supergraph.ingest.chunker import HeadingChunker
        chunker = HeadingChunker()
        chunks = chunker.chunk("# H\n" + "word " * 1000, max_chunk_size=200)
        assert len(chunks) > 1  # kwargs were honored

    def test_custom_chunker_satisfies_protocol(self):
        from supergraph.ingest.base import ChunkerProtocol
        from supergraph.ingest.base import Chunk

        class SingleChunker:
            def chunk(self, text: str, **kwargs):
                return [Chunk(text=text, summary=text[:50], index=0)]

        assert isinstance(SingleChunker(), ChunkerProtocol)

    def test_object_without_chunk_method_fails_protocol(self):
        from supergraph.ingest.base import ChunkerProtocol

        class NotAChunker:
            pass

        assert not isinstance(NotAChunker(), ChunkerProtocol)


class TestSuperGraphInjection:
    def test_custom_ingestor_used_for_extension(self, tmp_path):
        from supergraph import SuperGraph
        from supergraph.ingest.base import Ingestor, IngestResult

        called = []

        class TrackingIngestor(Ingestor):
            name = "tracker"
            supported_extensions = ["txt"]

            def convert(self, file_path: str, **kwargs) -> IngestResult:
                called.append(file_path)
                return IngestResult(markdown="# tracked\ncontent", parser_used="tracker")

        f = tmp_path / "notes.txt"
        f.write_text("# Hello\nworld")

        g = SuperGraph(path=str(tmp_path / "db"), embedder=None,
                       ingestors={"txt": TrackingIngestor()})
        g.execute(f'INGEST "{f}" AS "doc:t1"')
        g.close()

        assert len(called) == 1

    def test_custom_chunker_used(self, tmp_path):
        from supergraph import SuperGraph
        from supergraph.ingest.base import Chunk

        chunk_calls = []

        class CountingChunker:
            def chunk(self, text: str, **kwargs):
                chunk_calls.append(text)
                return [Chunk(text=text, summary=text[:50], index=0)]

        f = tmp_path / "doc.txt"
        f.write_text("# Title\nSome content here.")

        g = SuperGraph(path=str(tmp_path / "db"), embedder=None,
                       chunker=CountingChunker())
        g.execute(f'INGEST "{f}" AS "doc:c1"')
        g.close()

        assert len(chunk_calls) >= 1

    def test_default_path_preserved_without_ingestors(self, tmp_path):
        """No ingestors= passed → existing router path still active (no regression)."""
        from supergraph import SuperGraph

        f = tmp_path / "notes.txt"
        f.write_text("# Hello\nworld")

        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        result = g.execute(f'INGEST "{f}" AS "doc:default"')
        # router.py fast-paths .txt/.md as "direct"
        assert result.data["parser"] in ("markitdown", "direct")
        g.close()
