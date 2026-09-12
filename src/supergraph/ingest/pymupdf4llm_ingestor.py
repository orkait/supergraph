import logging
from supergraph.ingest.base import Ingestor, IngestResult, ExtractedImage
from supergraph.algos.chunker import chunk_by_heading, chunk_by_paragraph

logger = logging.getLogger(__name__)


class PyMuPDF4LLMIngestor(Ingestor):
    name = "pymupdf4llm"
    supported_extensions = ["pdf"]

    def convert(self, file_path: str, **kwargs) -> IngestResult:
        try:
            import pymupdf4llm
            import pymupdf
        except ImportError as e:
            raise ImportError(
                "PyMuPDF4LLMIngestor requires the `ingest` extra. "
                "Install with: pip install 'supergraph[ingest]'"
            ) from e

        page_chunks = pymupdf4llm.to_markdown(file_path, page_chunks=True)

        md_parts: list[str] = []
        chunks: list = []
        chunk_max = kwargs.get("max_chunk_size", 2000)
        summary_max = kwargs.get("summary_max_len", 200)
        overlap = kwargs.get("overlap", 50)
        for pc in page_chunks:
            page_num = int(pc.get("metadata", {}).get("page", 0)) + 1
            page_md = pc.get("text", "").strip()
            if not page_md:
                continue
            md_parts.append(page_md)
            page_chunks_out = chunk_by_heading(
                page_md,
                max_chunk_size=chunk_max,
                summary_max_len=summary_max,
                overlap=overlap,
            )
            if not page_chunks_out:
                page_chunks_out = chunk_by_paragraph(
                    page_md, max_chunk_size=chunk_max, summary_max_len=summary_max
                )
            for c in page_chunks_out:
                c.page = page_num
                c.index = len(chunks)
                chunks.append(c)

        md_text = "\n\n".join(md_parts)

        images = []
        doc = pymupdf.open(file_path)
        metadata = {"pages": len(doc), "source": file_path}
        if doc.metadata:
            metadata.update({k: v for k, v in doc.metadata.items() if v})

        seen_xrefs: set[int] = set()
        for page_num, page in enumerate(doc):
            for img_info in page.get_images(full=True):
                try:
                    xref = img_info[0]
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    base_image = doc.extract_image(xref)
                    if base_image:
                        images.append(ExtractedImage(
                            data=base_image["image"],
                            mime_type=f"image/{base_image['ext']}",
                            page=page_num + 1,
                        ))
                except Exception as e:
                    logger.debug("image extraction skipped for xref %s: %s", img_info[0], e, exc_info=True)
        doc.close()

        confidence = 1.0
        page_count = metadata.get("pages", 1) or 1
        chars_per_page = len(md_text) / page_count
        if chars_per_page < 50:
            confidence = 0.3
            metadata["warning"] = (
                "Low text extraction (likely scanned PDF). "
                "Consider: INGEST ... USING docling"
            )

        return IngestResult(
            markdown=md_text,
            chunks=chunks,
            images=images,
            metadata=metadata,
            parser_used=self.name,
            confidence=confidence,
        )
