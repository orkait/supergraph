"""Text chunking strategies - shim over supergraph.algos.chunker."""

from supergraph.algos.chunker import (
    Chunk,
    chunk_by_heading,
    chunk_by_paragraph,
    chunk_fixed,
    make_summary as _make_summary,
)

__all__ = [
    "Chunk",
    "chunk_by_heading",
    "chunk_by_paragraph",
    "chunk_fixed",
    "HeadingChunker",
]


class HeadingChunker:
    """Default ChunkerProtocol implementation - delegates to chunk_by_heading."""

    def chunk(self, text: str, **kwargs) -> list[Chunk]:
        return chunk_by_heading(text, **kwargs)
