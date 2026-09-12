
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
    paragraphs = _PARA_SPLIT_RE.split(text)
    chunks: list[Chunk] = []
    current = ""
    search_pos = 0
    current_start: int | None = None
    for para in paragraphs:
        para_stripped = para.strip()
        if not para_stripped:
            continue
        if current_start is None:
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
            search_pos = current_start + len(current.strip())
            current = ""
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
