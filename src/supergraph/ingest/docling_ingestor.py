from supergraph.ingest.base import Ingestor, IngestResult


class DoclingIngestor(Ingestor):
    name = "docling"
    supported_extensions = [
        "pdf", "docx", "pptx", "xlsx",
        "md", "html", "htm", "csv",
        "tex", "adoc",
        "png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp",
    ]

    def __init__(self) -> None:
        self._converter = None

    def _get_converter(self):
        if self._converter is None:
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
