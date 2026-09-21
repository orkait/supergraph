from __future__ import annotations

from superclaw.settings import (
    BRAILLE_BASE,
    BRAILLE_BITS,
    BRAILLE_COLS,
    BRAILLE_ROWS,
    LIMITS,
    RAMP_ASCII_RIGHT,
    RAMP_ASCII_SHADE,
    RAMP_ASCII_UP,
    RAMP_RIGHT,
    RAMP_SHADE,
    RAMP_UP,
    Glyphs,
)

UP, DOWN, LEFT, RIGHT = 1, 2, 4, 8
JOINS = {UP | DOWN: "│", LEFT | RIGHT: "─", DOWN | RIGHT: "╭", DOWN | LEFT: "╮",
         UP | RIGHT: "╰", UP | LEFT: "╯", UP | DOWN | RIGHT: "├", UP | DOWN | LEFT: "┤",
         DOWN | LEFT | RIGHT: "┬", UP | LEFT | RIGHT: "┴", UP | DOWN | LEFT | RIGHT: "┼",
         UP: "│", DOWN: "│", LEFT: "─", RIGHT: "─"}
ARROW_DOWN = "▼"
BRANCH, LAST, PIPE, BLANK = "├── ", "╰── ", "│   ", "    "
ASCII_BRANCH, ASCII_LAST, ASCII_PIPE = "|-- ", "`-- ", "|   "


def _ramp(glyphs: Glyphs, kind: str) -> str:
    if kind == "up":
        return RAMP_UP if glyphs.block_art else RAMP_ASCII_UP
    if kind == "right":
        return RAMP_RIGHT if glyphs.block_art else RAMP_ASCII_RIGHT
    return RAMP_SHADE if glyphs.block_art else RAMP_ASCII_SHADE


def _step(value: float, top: float, ramp: str) -> tuple[int, int]:
    share = 0.0 if top <= 0 else max(0.0, min(1.0, value / top))
    cells = share * (len(ramp) - 1)
    return int(cells), round((cells - int(cells)) * (len(ramp) - 1))


def bar(value: float, top: float, width: int, glyphs: Glyphs) -> str:
    ramp = _ramp(glyphs, "right")
    share = 0.0 if top <= 0 else max(0.0, min(1.0, value / top))
    cells = share * width
    full = int(cells)
    rest = int((cells - full) * (len(ramp) - 1))
    return (ramp[-1] * full + (ramp[rest] if rest else "")).ljust(width)


def bars(rows: list[tuple[str, float]], width: int, glyphs: Glyphs, unit: str = "") -> list[str]:
    if not rows:
        return []
    top = max(value for _, value in rows)
    label = max(len(name) for name, _ in rows)
    reading = max(len(f"{value:,.0f}{unit}") for _, value in rows)
    return [f"{name:<{label}}  {bar(value, top, width, glyphs)}  {f'{value:,.0f}{unit}':>{reading}}" for name, value in rows]


def gauge(used: float, total: float, width: int, glyphs: Glyphs) -> str:
    track = "░" if glyphs.block_art else "."
    return bar(used, total, width, glyphs).replace(" ", track)


def spark(series: list[float], glyphs: Glyphs) -> str:
    if not series:
        return ""
    ramp = _ramp(glyphs, "up")
    low, high = min(series), max(series)
    span = (high - low) or 1
    return "".join(ramp[1 + int((value - low) / span * (len(ramp) - 2))] for value in series)


def columns(series: list[float], height: int, glyphs: Glyphs) -> list[str]:
    if not series:
        return []
    ramp = _ramp(glyphs, "up")
    steps = len(ramp) - 1
    top = max(series) or 1
    rows = []
    for row in range(height, 0, -1):
        line = ""
        for value in series:
            filled = round(value / top * height * steps)
            here = filled - (row - 1) * steps
            line += ramp[-1] if here >= steps else (ramp[here] if here > 0 else " ")
        rows.append(line.rstrip() or " ")
    return rows


def curve(points: list[tuple[float, float]], width: int, height: int, glyphs: Glyphs) -> list[str]:
    if not glyphs.block_art:
        grid = [[" "] * width for _ in range(height)]
        for x, y in points:
            grid[min(height - 1, int((1 - y) * height))][min(width - 1, int(x * width))] = "*"
        return ["".join(row).rstrip() for row in grid]
    cells = [[0] * width for _ in range(height)]
    for x, y in points:
        dot_x = min(int(x * width * BRAILLE_COLS), width * BRAILLE_COLS - 1)
        dot_y = min(int((1 - y) * height * BRAILLE_ROWS), height * BRAILLE_ROWS - 1)
        cells[dot_y // BRAILLE_ROWS][dot_x // BRAILLE_COLS] |= BRAILLE_BITS[dot_y % BRAILLE_ROWS][dot_x % BRAILLE_COLS]
    return ["".join(chr(BRAILLE_BASE + cell) for cell in row) for row in cells]


def heat(matrix: list[list[float]], glyphs: Glyphs) -> list[str]:
    ramp = _ramp(glyphs, "shade")
    top = max((max(row) for row in matrix if row), default=0) or 1
    return ["".join(ramp[min(len(ramp) - 1, int(value / top * (len(ramp) - 0.001)))] * 2 for value in row) for row in matrix]


def tree(label: str, children: dict[str, list[str]], glyphs: Glyphs, depth: int = 2) -> list[str]:
    branch, last, pipe = (BRANCH, LAST, PIPE) if glyphs.block_art else (ASCII_BRANCH, ASCII_LAST, ASCII_PIPE)
    out = [label]

    def walk(node: str, prefix: str, level: int) -> None:
        kids = children.get(node, []) if level < depth else []
        for index, kid in enumerate(kids):
            final = index == len(kids) - 1
            out.append(f"{prefix}{last if final else branch}{kid}")
            walk(kid, prefix + (BLANK if final else pipe), level + 1)

    walk(label, "", 0)
    return out


class Grid:
    def __init__(self) -> None:
        self.chars: dict[tuple[int, int], str] = {}
        self.joins: dict[tuple[int, int], int] = {}

    def write(self, row: int, col: int, text: str) -> None:
        for offset, char in enumerate(text):
            self.chars[(row, col + offset)] = char

    def link(self, row: int, col: int, bits: int) -> None:
        self.joins[(row, col)] = self.joins.get((row, col), 0) | bits

    def across(self, row: int, start: int, end: int) -> None:
        low, high = min(start, end), max(start, end)
        for col in range(low, high + 1):
            self.link(row, col, (RIGHT if col < high else 0) | (LEFT if col > low else 0))

    def down(self, col: int, start: int, end: int) -> None:
        low, high = min(start, end), max(start, end)
        for row in range(low, high + 1):
            self.link(row, col, (DOWN if row < high else 0) | (UP if row > low else 0))

    def lines(self) -> list[str]:
        for spot, bits in self.joins.items():
            self.chars.setdefault(spot, JOINS.get(bits, "·"))
        if not self.chars:
            return []
        rows = [row for row, _ in self.chars]
        cols = [col for _, col in self.chars]
        return ["".join(self.chars.get((row, col), " ") for col in range(min(cols), max(cols) + 1)).rstrip()
                for row in range(min(rows), max(rows) + 1)]


def _box(grid: Grid, row: int, col: int, label: str, note: str) -> tuple[int, int, int]:
    inner = max(len(label), len(note)) + 2
    grid.write(row, col, "╭" + "─" * inner + "╮")
    grid.write(row + 1, col, "│ " + label.ljust(inner - 1) + "│")
    if note:
        grid.write(row + 2, col, "│ " + note.ljust(inner - 1) + "│")
    height = 3 if note else 2
    grid.write(row + height, col, "╰" + "─" * inner + "╯")
    return col + (inner + 2) // 2, height, inner + 2


def flow(layers: list[list[tuple[str, str]]], edges: list[tuple[str, str]], glyphs: Glyphs, gap: int = LIMITS.flow_gap) -> list[str]:
    grid = Grid()
    spans = [sum(max(len(label), len(note)) + 4 for label, note in layer) + LIMITS.flow_pad * (len(layer) - 1) for layer in layers]
    widest = max(spans) if spans else 0
    centres: dict[str, int] = {}
    edges_of: dict[str, tuple[int, int]] = {}
    row = 0
    for layer, span in zip(layers, spans, strict=True):
        col, tallest = (widest - span) // 2, 0
        for label, note in layer:
            centre, height, width = _box(grid, row, col, label, note)
            centres[label], edges_of[label] = centre, (row, row + height)
            col += width + LIMITS.flow_pad
            tallest = max(tallest, height)
        row += tallest + 1 + gap
    for source, target in edges:
        top, bottom = edges_of[source][1] + 1, edges_of[target][0] - 1
        bus = (top + bottom) // 2
        grid.down(centres[source], top, bus)
        grid.across(bus, centres[source], centres[target])
        grid.down(centres[target], bus, bottom)
        grid.chars[(bottom, centres[target])] = ARROW_DOWN
    return grid.lines()
