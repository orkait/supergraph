"""PyMuPDF4LLM ingestor: Tier 2 - PDF with structure + image extraction."""
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

        # page_chunks=True returns list[dict] with per-page text + metadata.
        # We keep the per-page structure so each Chunk can carry a `page`
        # number for citation use cases ("answer from page 14"). The
        # handler only re-chunks when IngestResult.chunks is empty; by
        # pre-chunking here we preserve provenance.
        page_chunks = pymupdf4llm.to_markdown(file_path, page_chunks=True)

        # Concatenate for the doc-level markdown blob (used by BM25 doc
        # storage). Separator is two newlines so downstream heading parse
        # does not lose block boundaries.
        md_parts: list[str] = []
        chunks: list = []
        chunk_max = kwargs.get("max_chunk_size", 2000)
        summary_max = kwargs.get("summary_max_len", 200)
        overlap = kwargs.get("overlap", 50)
        for pc in page_chunks:
            # pymupdf4llm uses "page" in metadata; 0-indexed. We expose
            # 1-indexed pages to users since that matches the reader UI.
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

        # Deduplicate by PDF xref. A single image (e.g. a header logo) used on
        # every page appears in every page's get_images() list with the same
        # xref. Pre-fix, a 50-page PDF with one logo stored 50 copies and
        # re-embedded the same image 50 times — bug #66. We record the first
        # page each unique image appears on (1-indexed to match chunk.page).
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

        # Scanned PDF detection
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
