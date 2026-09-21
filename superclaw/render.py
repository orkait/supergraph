from __future__ import annotations

import re
from dataclasses import dataclass

from rich.text import Text

from superclaw.settings import LIMITS, Glyphs
from superclaw.text import clip

FENCE = re.compile(r"^(`{3,}|~{3,})\s*([\w+-]*)")
HEADING = re.compile(r"^(#{1,6})\s+(.*)")
BULLET = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)")
QUOTE = re.compile(r"^\s*>\s?(.*)")
RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
ROW = re.compile(r"^\s*\|.*\|\s*$")
DIVIDER = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")
INLINE = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|__[^_]+__|(?<![\w*])\*[^*\n]+\*(?![\w*])|\[[^\]]+\]\([^)]+\))")
LINK = re.compile(r"^\[([^\]]+)\]\(([^)]+)\)$")
COLUMNS = re.compile(r"\S {2,}\S")
DRAWING = (("─", "╿"), ("▀", "▟"), ("⠀", "⣿"), ("■", "◿"), ("←", "⇿"))
CODE_STYLE, BOLD, ITALIC, LINK_STYLE, MUTED, HEAD = "bold", "bold", "italic", "underline", "dim", "bold"
RIGHT, CENTRE = "right", "centre"


def drawn(line: str) -> bool:
    return any(low <= ch <= high for ch in line for low, high in DRAWING)


def preformatted(lines: list[str]) -> bool:
    if any(drawn(line) for line in lines):
        return True
    if len(lines) < 2:
        return False
    aligned = sum(bool(COLUMNS.search(line)) for line in lines)
    indented = sum(line.startswith(" " * LIMITS.preformatted_indent) for line in lines)
    return aligned >= 2 or indented * 2 >= len(lines)


@dataclass(frozen=True)
class Word:
    text: str
    style: str


def segments(line: str) -> list[Word]:
    out: list[Word] = []
    for piece in INLINE.split(line):
        if not piece:
            continue
        if piece.startswith("`") and piece.endswith("`"):
            out.append(Word(piece[1:-1], CODE_STYLE))
        elif (piece.startswith("**") and piece.endswith("**")) or (piece.startswith("__") and piece.endswith("__")):
            out.append(Word(piece[2:-2], BOLD))
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            out.append(Word(piece[1:-1], ITALIC))
        elif (link := LINK.match(piece)) is not None:
            out.append(Word(link.group(1), LINK_STYLE))
        else:
            out.append(Word(piece, ""))
    return out


def words_of(parts: list[Word]) -> list[Word]:
    out: list[Word] = []
    for part in parts:
        out.extend(Word(piece, part.style) for piece in re.split(r"(\s+)", part.text) if piece)
    return out


def wrap(parts: list[Word], measure: int, first: str = "", later: str = "") -> list[Text]:
    lines: list[Text] = []
    line, width, prefix = Text(first), len(first), later or first
    for word in words_of(parts):
        blank = not word.text.strip()
        size = len(word.text)
        if blank and width == len(prefix):
            continue
        if width + size > measure and not blank and width > len(prefix):
            lines.append(line)
            line, width = Text(prefix), len(prefix)
        line.append(word.text, style=word.style or None)
        width += size
    lines.append(line)
    return [entry for entry in lines if entry.plain.strip() or len(lines) == 1]


def code(lines: list[str], language: str, measure: int, glyphs: Glyphs) -> list[Text]:
    body = Text("\n".join(lines))
    if language:
        try:
            from rich.syntax import Syntax

            body = Syntax(code="", lexer=language, theme="ansi_dark").highlight("\n".join(lines))
        except Exception:
            body = Text("\n".join(lines))
    out = []
    if language:
        out.append(Text(f"{glyphs.bar} {clip(language, LIMITS.code_label_chars)}", style=MUTED))
    for row in body.split("\n")[: len(lines)]:
        row.truncate(max(1, measure - 2), overflow="ellipsis")
        out.append(Text(f"{glyphs.bar} ", style=MUTED) + row)
    return out


def verbatim(lines: list[str], measure: int) -> list[Text]:
    out = []
    for line in lines:
        row = Text(line)
        row.truncate(measure, overflow="ellipsis")
        out.append(row)
    return out


def cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def alignments(divider: str) -> list[str]:
    return [CENTRE if cell.startswith(":") and cell.endswith(":") else RIGHT if cell.endswith(":") else "" for cell in cells(divider)]


def widths(rows: list[list[str]], measure: int, glyphs: Glyphs) -> list[int]:
    columns = max(len(row) for row in rows)
    found = [max((len(row[i]) for row in rows if i < len(row)), default=0) for i in range(columns)]
    separator = len(f" {glyphs.pipe} ")
    while sum(found) + separator * (columns - 1) > measure and max(found) > LIMITS.table_min_column:
        found[found.index(max(found))] -= 1
    return found


def cut(value: str, width: int, align: str) -> str:
    text = clip(value, width)
    if align == RIGHT:
        return text.rjust(width)
    if align == CENTRE:
        return text.center(width)
    return text.ljust(width)


def stack(cell: str, width: int, align: str) -> list[str]:
    plain = [line.plain for line in wrap(segments(cell), width)] or [""]
    return [cut(line, width, align) for line in plain]


def table(rows: list[list[str]], aligns: list[str], measure: int, glyphs: Glyphs) -> list[Text]:
    sizes = widths(rows, measure, glyphs)
    gap = f" {glyphs.pipe} "
    rule = Text(f"{glyphs.rule}{glyphs.cross}{glyphs.rule}".join(glyphs.rule * size for size in sizes), style=MUTED)
    stacked = [[stack(row[i] if i < len(row) else "", sizes[i], aligns[i] if i < len(aligns) else "") for i in range(len(sizes))] for row in rows]
    tall = any(len(cell) > 1 for row in stacked for cell in row)
    out: list[Text] = []
    for index, row in enumerate(stacked):
        for line in range(max(len(cell) for cell in row)):
            built = Text(gap, style=MUTED).join(Text(cell[line] if line < len(cell) else " " * len(cell[0]),
                                                     style=HEAD if index == 0 else None) for cell in row)
            built.rstrip()
            out.append(built)
        if index == 0 or (tall and index < len(stacked) - 1):
            out.append(rule)
    return out


def blocks(text: str) -> list[list[str]]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line)
        elif current:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def paragraph(lines: list[str], measure: int, glyphs: Glyphs) -> list[Text]:
    if preformatted(lines):
        return verbatim(lines, measure)
    return wrap(segments(" ".join(line.strip() for line in lines)), measure)


def listing(lines: list[str], measure: int, glyphs: Glyphs) -> list[Text]:
    out: list[Text] = []
    for line in lines:
        match = BULLET.match(line)
        if match is None:
            out += wrap(segments(line.strip()), measure, first="  ")
            continue
        indent, marker, body = match.group(1), match.group(2), match.group(3)
        head = f"{indent}{glyphs.bullet} " if marker in "-*+" else f"{indent}{marker} "
        out += wrap(segments(body), measure, first=head, later=" " * len(head))
    return out


def block(lines: list[str], prose: int, measure: int, glyphs: Glyphs) -> list[Text]:
    head = lines[0].strip()
    if (title := HEADING.match(head)) is not None:
        rest = block(lines[1:], prose, measure, glyphs) if len(lines) > 1 else []
        return [Text(title.group(2), style=HEAD), *rest]
    if RULE.match(head) is not None:
        return [Text(glyphs.rule * min(prose, measure), style=MUTED)]
    if len(lines) > 1 and ROW.match(lines[0]) and DIVIDER.match(lines[1]):
        rows = [cells(line) for line in lines if ROW.match(line) and not DIVIDER.match(line)]
        return table(rows, alignments(lines[1]), measure, glyphs)
    if all(QUOTE.match(line) for line in lines):
        body = wrap(segments(" ".join(QUOTE.match(line).group(1) for line in lines)), prose - 2)
        return [Text(f"{glyphs.bar} ", style=MUTED) + line for line in body]
    if BULLET.match(lines[0]) is not None:
        return listing(lines, prose, glyphs)
    return paragraph(lines, prose if not preformatted(lines) else measure, glyphs)


def settled_at(text: str) -> int:
    lines = text.split("\n")
    cut, opened, fenced, position = 0, -1, False, 0
    for index, line in enumerate(lines):
        if FENCE.match(line.strip()) is not None:
            fenced = not fenced
            opened = position if fenced else -1
        elif not fenced and not line.strip() and index < len(lines) - 1:
            cut = position + len(line) + 1
        position += len(line) + 1
    return min(cut, opened) if opened >= 0 else cut


class Stream:
    def __init__(self, prose: int, measure: int, glyphs: Glyphs) -> None:
        self.prose, self.measure, self.glyphs = prose, measure, glyphs
        self.settled, self.consumed = Text(), 0

    def update(self, text: str) -> Text:
        cut = settled_at(text)
        if cut > self.consumed:
            done = markdown(text[self.consumed:cut], self.prose, self.measure, self.glyphs)
            self.settled = Text("\n\n").join([part for part in (self.settled, done) if part.plain])
            self.consumed = cut
        tail = markdown(text[self.consumed:], self.prose, self.measure, self.glyphs)
        return Text("\n\n").join([part for part in (self.settled, tail) if part.plain])


def markdown(text: str, prose: int, measure: int, glyphs: Glyphs) -> Text:
    prose = max(LIMITS.table_min_column, min(prose, measure))
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[Text] = []
    index = 0
    while index < len(lines):
        opening = FENCE.match(lines[index].strip())
        if opening is not None:
            language, body = opening.group(2), []
            index += 1
            while index < len(lines) and not FENCE.match(lines[index].strip()):
                body.append(lines[index])
                index += 1
            index += 1
            if out and out[-1].plain:
                out.append(Text())
            out += code(body, language, measure, glyphs)
            out.append(Text())
            continue
        if not lines[index].strip():
            index += 1
            continue
        group = []
        while index < len(lines) and lines[index].strip() and not FENCE.match(lines[index].strip()):
            group.append(lines[index])
            index += 1
        if out and out[-1].plain:
            out.append(Text())
        out += block(group, prose, measure, glyphs)
    while out and not out[-1].plain:
        out.pop()
    return Text("\n").join(out)
