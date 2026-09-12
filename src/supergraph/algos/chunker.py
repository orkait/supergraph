"""Pure text chunking primitives for ingestion.

Splits markdown / plaintext into Chunk records. No supergraph imports,
no I/O. Takes a string, returns a list of Chunks.
"""

import re
from dataclasses import dataclass

__all__ = [
    "Chunk",
    "make_summary",
    "chunk_by_heading",
    "chunk_by_paragraph",
    "chunk_fixed",
]


@dataclass
class Chunk:
    text: str
    summary: str
    index: int
    heading: str | None = None
    page: int | None = None
    start_char: int = 0


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")


def make_summary(text: str, max_len: int = 200) -> str:
    s = text[:max_len].strip()
    if len(text) > max_len:
        s = s.rsplit(" ", 1)[0] + "..."
    return s


def chunk_fixed(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    summary_max_len: int = 200,
) -> list[Chunk]:
    """Fixed-size sliding window chunks with overlap."""
    # Guard against infinite loops. ``pos += chunk_size - overlap`` never
    # advances when chunk_size == overlap (step of 0) and regresses when
    # overlap > chunk_size (negative step), making the outer while-loop
    # infinite on any non-empty input. Pre-fix, a caller with a bad config
    # (e.g. chunk_max_size=50, chunk_overlap=50 from a misconfigured JSON)
    # could wedge ingest indefinitely — bug #83. Reject at entry.
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if chunk_size <= overlap:
        raise ValueError(
            f"chunk_size must exceed overlap "
            f"(got chunk_size={chunk_size}, overlap={overlap})"
        )
    chunks: list[Chunk] = []
    pos = 0
    while pos < len(text):
        end = min(pos + chunk_size, len(text))
        chunk_text = text[pos:end]
        chunks.append(
            Chunk(
                text=chunk_text,
                summary=make_summary(chunk_text, summary_max_len),
                index=len(chunks),
                start_char=pos,
            )
        )
        pos += chunk_size - overlap
        if pos >= len(text):
            break
    return chunks


def chunk_by_paragraph(
    text: str,
    max_chunk_size: int = 1000,
    summary_max_len: int = 200,
) -> list[Chunk]:
    """Split on double newlines, packing paragraphs up to max_chunk_size."""
    paragraphs = _PARA_SPLIT_RE.split(text)
    chunks: list[Chunk] = []
    current = ""
    # Track where the current chunk starts in the ORIGINAL text, not where
    # the accumulator thinks it is. Pre-fix, start_char advanced by
    # len(current) which excluded the whitespace eaten by _PARA_SPLIT_RE
    # and the .strip() calls, drifting a few characters per chunk (bug
    # #84). We use str.find() from the previous chunk's end so the offset
    # is always the real byte position of the chunk's first non-whitespace
    # character.
    search_pos = 0
    current_start: int | None = None
    for para in paragraphs:
        para_stripped = para.strip()
        if not para_stripped:
            continue
        if current_start is None:
            # Locate the start of this paragraph in the original text.
            loc = text.find(para_stripped, search_pos)
            current_start = loc if loc >= 0 else search_pos
        if current and (
            len(current) + len(para_stripped) > max_chunk_size
            or len(current.strip()) >= max_chunk_size // 2
        ):
            chunks.append(
                Chunk(
                    text=current.strip(),
                    summary=make_summary(current.strip(), summary_max_len),
                    index=len(chunks),
                    start_char=current_start,
                )
            )
            # Advance search_pos so the next find() starts after the chunk
            # we just emitted, and reset current_start so the next iteration
            # locates the new chunk's real start.
            search_pos = current_start + len(current.strip())
            current = ""
            # Locate THIS paragraph's start for the new chunk.
            loc = text.find(para_stripped, search_pos)
            current_start = loc if loc >= 0 else search_pos
        current += para_stripped + "\n\n"
    if current.strip():
        chunks.append(
            Chunk(
                text=current.strip(),
                summary=make_summary(current.strip(), summary_max_len),
                index=len(chunks),
                start_char=current_start if current_start is not None else 0,
            )
        )
    if not chunks:
        chunks = [
            Chunk(
                text=text.strip(),
                summary=make_summary(text.strip(), summary_max_len),
                index=0,
                start_char=0,
            )
        ]
    return chunks


def chunk_by_heading(
    text: str,
    max_chunk_size: int = 2000,
    summary_max_len: int = 200,
    overlap: int = 50,
) -> list[Chunk]:
    """Split on markdown headings; fall back to paragraph split if none found."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return chunk_by_paragraph(text, max_chunk_size, summary_max_len=summary_max_len)

    chunks: list[Chunk] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            chunks.append(
                Chunk(
                    text=preamble,
                    summary=make_summary(preamble, summary_max_len),
                    index=0,
                    start_char=0,
                )
            )

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        heading = match.group(2).strip()

        if len(section) > max_chunk_size:
            sub_chunks = chunk_fixed(
                section,
                chunk_size=max_chunk_size,
                overlap=overlap,
                summary_max_len=summary_max_len,
            )
            for sc in sub_chunks:
                sc.heading = heading
                sc.index = len(chunks)
                sc.start_char = start + sc.start_char
                chunks.append(sc)
        else:
            chunks.append(
                Chunk(
                    text=section,
                    summary=make_summary(section, summary_max_len),
                    index=len(chunks),
                    heading=heading,
                    start_char=start,
                )
            )

    for i, c in enumerate(chunks):
        c.index = i
    return chunks
