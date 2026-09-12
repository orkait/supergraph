import pytest
from supergraph.ingest.base import IngestResult, Ingestor, ExtractedImage
from supergraph.ingest.chunker import chunk_by_heading, chunk_by_paragraph, chunk_fixed, _make_summary


class TestMakeSummary:
    def test_short_text_unchanged(self):
        assert _make_summary("hello") == "hello"

    def test_long_text_truncated(self):
        s = _make_summary("word " * 100, max_len=50)
        assert len(s) <= 55
        assert s.endswith("...")

    def test_empty_text(self):
        assert _make_summary("") == ""


class TestChunkByHeading:
    def test_splits_on_headings(self):
        text = "# Intro\nHello world\n## Details\nMore stuff\n## End\nBye"
        chunks = chunk_by_heading(text)
        assert len(chunks) >= 3
        assert chunks[0].heading == "Intro"
        assert "Hello world" in chunks[0].text

    def test_no_headings_falls_back(self):
        text = "Just plain text without any headings."
        chunks = chunk_by_heading(text)
        assert len(chunks) == 1

    def test_chunk_has_summary(self):
        text = "# Overview\nThis is a detailed section about something important."
        chunks = chunk_by_heading(text)
        assert chunks[0].summary
        assert len(chunks[0].summary) <= 203

    def test_preamble_before_first_heading(self):
        text = "Some preamble text\n\n# Heading\nContent"
        chunks = chunk_by_heading(text)
        assert chunks[0].heading is None
        assert "preamble" in chunks[0].text

    def test_respects_max_size(self):
        text = "# Big Section\n" + "word " * 1000
        chunks = chunk_by_heading(text, max_chunk_size=200)
        assert len(chunks) > 1


class TestChunkByParagraph:
    def test_splits_on_double_newline(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_by_paragraph(text, max_chunk_size=50)
        assert len(chunks) >= 2

    def test_single_paragraph(self):
        text = "Just one paragraph."
        chunks = chunk_by_paragraph(text)
        assert len(chunks) == 1

    def test_empty_text(self):
        chunks = chunk_by_paragraph("")
        assert len(chunks) == 1


class TestChunkFixed:
    def test_fixed_size(self):
        text = "a" * 500
        chunks = chunk_fixed(text, chunk_size=100, overlap=20)
        assert len(chunks) >= 5

    def test_overlap(self):
        text = "abcdefghij" * 50
        chunks = chunk_fixed(text, chunk_size=100, overlap=20)
        if len(chunks) > 1:
            end_of_first = chunks[0].text[-20:]
            start_of_second = chunks[1].text[:20]
            assert end_of_first == start_of_second

    def test_each_chunk_has_summary(self):
        text = "x" * 500
        chunks = chunk_fixed(text, chunk_size=100)
        for c in chunks:
            assert c.summary


class TestBaseProtocol:
    def test_ingest_result_defaults(self):
        r = IngestResult(markdown="hello")
        assert r.chunks == []
        assert r.images == []
        assert r.confidence == 1.0

    def test_extracted_image(self):
        img = ExtractedImage(data=b"png", mime_type="image/png", page=3)
        assert img.page == 3

    def test_ingestor_not_implemented(self):
        import pytest
        with pytest.raises(NotImplementedError):
            Ingestor().convert("test.txt")


class TestMarkItDownIngestor:
    def test_convert_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world\nThis is a test")
        from supergraph.ingest.markitdown_ingestor import MarkItDownIngestor
        ingestor = MarkItDownIngestor()
        result = ingestor.convert(str(f))
        assert "Hello" in result.markdown
        assert result.parser_used == "markitdown"

    def test_supported_extensions(self):
        from supergraph.ingest.markitdown_ingestor import MarkItDownIngestor
        assert "txt" in MarkItDownIngestor.supported_extensions
        assert "html" in MarkItDownIngestor.supported_extensions


class TestRouter:
    def test_txt_routes_to_markitdown(self):
        from supergraph.ingest.router import select_ingestor
        assert select_ingestor("notes.txt") == "markitdown"

    def test_pdf_routes_to_pymupdf4llm(self):
        from supergraph.ingest.router import select_ingestor
        assert select_ingestor("report.pdf") == "pymupdf4llm"

    def test_docx_routes_to_markitdown(self):
        from supergraph.ingest.router import select_ingestor
        assert select_ingestor("doc.docx") == "markitdown"

    def test_explicit_override(self):
        from supergraph.ingest.router import select_ingestor
        assert select_ingestor("report.pdf", using="docling") == "docling"

    def test_unsupported_format_raises(self):
        import pytest
        from supergraph.ingest.router import select_ingestor
        with pytest.raises(ValueError, match="Unsupported format"):
            select_ingestor("file.xyz")

    def test_list_ingestors(self):
        from supergraph.ingest.router import list_ingestors
        ingestors = list_ingestors()
        names = [i["name"] for i in ingestors]
        assert "markitdown" in names
        assert "pymupdf4llm" in names
        assert "docling" in names
        assert "vision" in names
        assert "audio" not in names
        for ing in ingestors:
            assert "extra" in ing, f"{ing['name']} missing 'extra' field"


class TestRouterDoclingExtensions:

    @pytest.fixture(autouse=True)
    def require_docling(self):
        pytest.importorskip("docling")

    def test_tex_routes_to_docling(self):
        from supergraph.ingest.router import select_ingestor
        assert select_ingestor("paper.tex") == "docling"

    def test_adoc_routes_to_docling(self):
        from supergraph.ingest.router import select_ingestor
        assert select_ingestor("manual.adoc") == "docling"

    def test_tiff_routes_to_docling(self):
        from supergraph.ingest.router import select_ingestor
        assert select_ingestor("scan.tiff") == "docling"

    def test_tif_routes_to_docling(self):
        from supergraph.ingest.router import select_ingestor
        assert select_ingestor("scan.tif") == "docling"

    def test_bmp_routes_to_docling(self):
        from supergraph.ingest.router import select_ingestor
        assert select_ingestor("image.bmp") == "docling"

    def test_m4a_routes_to_docling(self):
        from supergraph.ingest.router import select_ingestor
        assert select_ingestor("audio.m4a") == "docling"

    def test_aac_routes_to_docling(self):
        from supergraph.ingest.router import select_ingestor
        assert select_ingestor("audio.aac") == "docling"

    def test_mp4_routes_to_docling(self):
        from supergraph.ingest.router import select_ingestor
        assert select_ingestor("video.mp4") == "docling"

    def test_avi_routes_to_docling(self):
        from supergraph.ingest.router import select_ingestor
        assert select_ingestor("video.avi") == "docling"

    def test_mov_routes_to_docling(self):
        from supergraph.ingest.router import select_ingestor
        assert select_ingestor("video.mov") == "docling"

    def test_docling_ingestor_supported_extensions_complete(self):
        from supergraph.ingest.docling_ingestor import DoclingIngestor
        exts = set(DoclingIngestor.supported_extensions)
        assert "tex" in exts
        assert "adoc" in exts
        assert "tiff" in exts
        assert "bmp" in exts
        assert "m4a" in exts
        assert "mp4" in exts

    def test_existing_formats_unchanged(self):
        from supergraph.ingest.router import select_ingestor
        assert select_ingestor("report.pdf") == "pymupdf4llm"
        assert select_ingestor("doc.docx") == "markitdown"
        assert select_ingestor("notes.txt") == "markitdown"

    def test_ingest_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("# Hello\n\nWorld")
        from supergraph.ingest.router import ingest_file
        result = ingest_file(str(f))
        assert "Hello" in result.markdown


class TestVisionHandler:
    def test_init_with_explicit_base_url(self):
        from supergraph.ingest.vision import VisionHandler
        vh = VisionHandler(base_url="http://example.invalid:9/v1")
        assert vh.model == "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf"
        assert vh._base_url == "http://example.invalid:9/v1"

    def test_init_resolves_via_env(self, monkeypatch):
        from supergraph.ingest.vision import VisionHandler
        monkeypatch.setenv("SUPERGRAPH_VISION_URL", "http://remote.invalid:1234/v1")
        vh = VisionHandler()
        assert vh._base_url == "http://remote.invalid:1234/v1"


from supergraph import SuperGraph


class TestIngestDSL:
    def test_ingest_text_file(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("# Meeting Notes\n\nDiscussed Q3 results.\n\n## Action Items\n\nFollow up with Alice.")
        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        result = g.execute(f'INGEST "{f}"')
        assert result.data["chunks"] >= 2
        assert result.data["parser"] in ("markitdown", "direct")
        g.close()

    def test_ingest_with_as_and_kind(self, tmp_path):
        f = tmp_path / "report.txt"
        f.write_text("# Report\nContent here")
        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        result = g.execute(f'INGEST "{f}" AS "doc:report" KIND "report"')
        assert result.data["doc_id"] == "doc:report"
        node = g.execute('NODE "doc:report"')
        assert node.data is not None
        assert node.data["kind"] == "report"
        g.close()

    def test_ingest_creates_edges(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("# Part 1\nContent\n\n# Part 2\nMore content")
        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        g.execute(f'INGEST "{f}" AS "doc:test"')
        edges = g.execute('EDGES FROM "doc:test"')
        assert len(edges.data) >= 2
        g.close()

    def test_node_with_document(self, tmp_path):
        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        g.execute('CREATE NODE "n1" kind = "note" name = "test" DOCUMENT "Full text here"')
        node = g.execute('NODE "n1"')
        assert "_document" not in node.data
        node = g.execute('NODE "n1" WITH DOCUMENT')
        assert node.data["_document"] == "Full text here"
        g.close()

    def test_ingest_nonexistent_raises(self, tmp_path):
        import pytest
        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        with pytest.raises(Exception):
            g.execute('INGEST "/nonexistent/file.txt"')
        g.close()

    def test_ingest_duplicate_raises(self, tmp_path):
        import pytest
        f = tmp_path / "dup.txt"
        f.write_text("content")
        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        g.execute(f'INGEST "{f}" AS "doc:dup"')
        with pytest.raises(Exception):
            g.execute(f'INGEST "{f}" AS "doc:dup"')
        g.close()

    def test_chunks_have_summaries(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("# Section 1\nSome detailed content about something important.")
        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        g.execute(f'INGEST "{f}" AS "doc:sum"')
        nodes = g.execute('NODES WHERE kind = "chunk"')
        assert len(nodes.data) >= 1
        assert "summary" in nodes.data[0]
        g.close()

    def test_ingest_stores_document_in_docstore(self, tmp_path):
        f = tmp_path / "stored.txt"
        f.write_text("# Hello\nThis is the full document content.")
        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        g.execute(f'INGEST "{f}" AS "doc:stored"')
        node = g.execute('NODE "doc:stored" WITH DOCUMENT')
        assert node.data["_document_type"] == "text/markdown"
        assert "Hello" in node.data["_document"]
        g.close()

    def test_ingest_auto_id(self, tmp_path):
        f = tmp_path / "auto.txt"
        f.write_text("# Auto ID\nContent")
        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        result = g.execute(f'INGEST "{f}"')
        assert result.data["doc_id"].startswith("doc:")
        assert len(result.data["doc_id"]) > 4
        g.close()


class TestIngestSecurity:
    def test_path_traversal_blocked_by_ingest_root(self, tmp_path):
        import pytest
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        (allowed_dir / "safe.txt").write_text("safe content")

        outside_file = tmp_path / "secret.txt"
        outside_file.write_text("secret data")

        g = SuperGraph(path=str(tmp_path / "db"), embedder=None, ingest_root=str(allowed_dir))

        result = g.execute(f'INGEST "{allowed_dir / "safe.txt"}" AS "doc:safe"')
        assert result.data["chunks"] >= 1

        with pytest.raises(Exception, match="Path traversal not allowed"):
            g.execute(f'INGEST "{outside_file}" AS "doc:secret"')

        with pytest.raises(Exception, match="Path traversal not allowed"):
            g.execute('INGEST "/etc/passwd" AS "doc:passwd"')

        g.close()

    def test_no_ingest_root_allows_any_path(self, tmp_path):
        f = tmp_path / "anywhere.txt"
        f.write_text("content")
        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        result = g.execute(f'INGEST "{f}" AS "doc:any"')
        assert result.data["chunks"] >= 1
        g.close()
