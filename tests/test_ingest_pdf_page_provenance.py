"""PDF ingest page provenance + warning surfacing.

Runs only when pymupdf + pymupdf4llm are installed (needs_ingest marker).
"""
import pytest

pytestmark = pytest.mark.needs_ingest


def _make_pdf(path, pages_text: list[str]) -> None:
    """Build a tiny synthetic PDF with one paragraph per page."""
    import pymupdf
    doc = pymupdf.open()
    for txt in pages_text:
        page = doc.new_page()
        page.insert_text((72, 100), txt)
    doc.save(str(path))
    doc.close()


def test_pdf_chunks_carry_page_number(tmp_path):
    from supergraph import SuperGraph
    pdf = tmp_path / "three.pdf"
    _make_pdf(pdf, [
        "Alpha content on page one.",
        "Beta content on page two.",
        "Gamma content on page three.",
    ])

    gs = SuperGraph(path=str(tmp_path / "gs"), embedder=None, ingest_root=str(tmp_path))
    try:
        r = gs.execute(f'INGEST "{pdf}" AS "doc:t"')
        assert r.data["parser"] == "pymupdf4llm"
        # Every chunk should have a page number that is 1-indexed.
        chunks = gs.execute('NODES WHERE kind = "chunk"').data
        assert chunks, "no chunks created"
        pages = [c.get("page") for c in chunks]
        assert all(p is not None for p in pages), (
            f"expected every chunk to carry a page; got {pages!r}"
        )
        assert min(pages) >= 1 and max(pages) <= 3, (
            f"page numbers out of range: {pages!r}"
        )
    finally:
        gs.close()


def test_scanned_pdf_surfaces_warning(tmp_path):
    """PDFs with <50 chars per page should surface a warning in meta."""
    from supergraph import SuperGraph
    pdf = tmp_path / "scanned.pdf"
    # Tiny text per page simulates a scanned PDF where OCR would be needed.
    _make_pdf(pdf, ["x", "x", "x", "x"])

    gs = SuperGraph(path=str(tmp_path / "gs2"), embedder=None, ingest_root=str(tmp_path))
    try:
        r = gs.execute(f'INGEST "{pdf}" AS "doc:scan"')
        warnings = (r.meta or {}).get("warnings") or []
        assert warnings, f"expected warnings in meta; got meta={r.meta!r}"
        assert any("scanned" in w.lower() or "confidence" in w.lower() for w in warnings)
        # The ingest must still succeed (soft warning, not a hard failure)
        assert r.data["doc_id"] == "doc:scan"
    finally:
        gs.close()
