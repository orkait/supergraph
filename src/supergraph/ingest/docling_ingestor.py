"""Docling ingestor: Tier 3 - hard PDFs with tables, OCR, formulas. Lazy import."""
from supergraph.ingest.base import Ingestor, IngestResult


class DoclingIngestor(Ingestor):
    name = "docling"
    supported_extensions = [
        "pdf", "docx", "pptx", "xlsx",                            # office / documents
        "md", "html", "htm", "csv",                               # markup / structured
        "tex", "adoc",                                             # LaTeX, AsciiDoc
        "png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp",       # images
    ]
    # Audio/video (m4a, aac, mp4, avi, mov) were advertised but supergraph has
    # no test exercising docling[asr]. Removed to match tested surface.
    # Users who need ASR can run ``pip install docling[asr]`` and call
    # docling directly, then ``gs.execute(... DOCUMENT ...)`` with the result.

    def __init__(self) -> None:
        # Lazy converter: constructing ``DocumentConverter`` costs ~200 MB of
        # model weights and a few seconds of load time. Pre-fix, every
        # convert() call reconstructed a fresh instance which made bulk PDF
        # ingestion catastrophically slow (bug #67). We cache the instance on
        # the ingestor itself and share across calls.
        self._converter = None

    def _get_converter(self):
        if self._converter is None:
            # Deferred import keeps the 200 MB dependency optional for users
            # who only ingest via MarkItDown / PyMuPDF.
            from docling.document_converter import DocumentConverter
            self._converter = DocumentConverter()
        return self._converter

    def convert(self, file_path: str, **kwargs) -> IngestResult:
        converter = self._get_converter()
        result = converter.convert(file_path)
        md_text = result.document.export_to_markdown()
        return IngestResult(
            markdown=md_text,
            metadata={"source": file_path},
            parser_used=self.name,
        )
