from __future__ import annotations

import bisect
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise

from superclaw.runtime import approx_tokens
from superclaw.settings import LIMITS


class Category(str, Enum):
    DEFAULT = "default"
    FILE = "file"
    SEARCH = "search"
    TEST = "test"
    PROCESS = "process"
    DIFF = "diff"


@dataclass(frozen=True)
class Budget:
    max_tokens: int
    max_chars: int

    def fits(self, tokens: int, chars: int) -> bool:
        return tokens <= self.max_tokens and chars <= self.max_chars


@dataclass
class Budgeted:
    text: str
    category: Category
    original_chars: int
    original_tokens: int
    truncated: bool = False
    reason: str = ""

    @property
    def retained_chars(self) -> int:
        return len(self.text)

    @property
    def retained_tokens(self) -> int:
        return approx_tokens(self.text)


OMISSION = "[... {count} lines omitted ...]"
_SEARCH_LOCATION = re.compile(r"^([^:\s][^:]*):\d+[:-]")
_TEST_FAILURE = re.compile(r"(?i)\b(failed|error|assert|traceback|panic|exception)\b|^FAIL\b|^E\s")
_PROCESS_DIAGNOSTIC = re.compile(r"(?i)\b(error|warning|warn:|fatal|panic|failed|denied)\b")
_DIFF_HEADER = re.compile(r"^(diff --git|--- |\+\+\+ |index |@@ )")


def budget_for(category: Category) -> Budget:
    tokens = (LIMITS.read_file_tokens if category is Category.FILE else LIMITS.tool_output_tokens) - LIMITS.truncation_notice_tokens
    return Budget(max_tokens=tokens, max_chars=tokens * LIMITS.chars_per_token)


def budget_output(text: str, category: Category, budget: Budget | None = None) -> Budgeted:
    budget = budget or budget_for(category)
    original_tokens = approx_tokens(text)
    result = Budgeted(text, category, len(text), original_tokens)
    if budget.fits(original_tokens, len(text)):
        return result
    units = _units(text, category)
    priorities = _PRIORITIES[category](units)
    retained = _render(units, _select(units, priorities, budget))
    if retained.strip() and budget.fits(approx_tokens(retained), len(retained)):
        result.text, result.reason = retained, f"semantic_{category.value}_budget"
    else:
        result.text, result.reason = _head_tail(text, budget), "head_tail_budget"
    result.truncated = True
    return result


def _units(text: str, category: Category) -> list[str]:
    lines = text.split("\n")
    if category is not Category.DIFF:
        return _collapse_repeats(lines)
    units: list[str] = []
    current: list[str] = []
    for line in lines:
        if current and (_DIFF_HEADER.match(line) or line.startswith("@@")):
            units.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        units.append("\n".join(current))
    return units


def _collapse_repeats(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        if not out or out[-1] != line:
            out.append(line)
    return out


def _sequence(n: int) -> list[int]:
    return list(range(n))


def _tail(n: int) -> list[int]:
    return list(range(n - 1, max(-1, n - 1 - LIMITS.budget_tail_lines), -1))


def _edges(n: int, head: int, tail: int) -> list[int]:
    return [*range(min(head, n)), *range(max(0, n - tail), n)]


def _file_priorities(units: list[str]) -> list[int]:
    priorities = [0] if units else []
    left, right = 1, len(units) - 1
    while left <= right:
        priorities.append(left)
        if right != left:
            priorities.append(right)
        left, right = left + 1, right - 1
    return priorities


def _search_priorities(units: list[str]) -> list[int]:
    priorities = [i for i, line in enumerate(units) if not _SEARCH_LOCATION.match(line)]
    seen: set[str] = set()
    for i, line in enumerate(units):
        match = _SEARCH_LOCATION.match(line)
        if match and match.group(1) not in seen:
            seen.add(match.group(1))
            priorities.append(i)
    return [*priorities, *_sequence(len(units))]


def _test_priorities(units: list[str]) -> list[int]:
    priorities: list[int] = []
    for i, line in enumerate(units):
        if _TEST_FAILURE.search(line):
            priorities.extend(range(max(0, i - LIMITS.budget_failure_context_before), min(len(units), i + LIMITS.budget_failure_context_after + 1)))
    return [*priorities, *_tail(len(units)), *range(min(LIMITS.budget_head_lines, len(units))), *_sequence(len(units))]


def _process_priorities(units: list[str]) -> list[int]:
    seen: set[str] = set()
    diagnostics = []
    for i, line in enumerate(units):
        if _PROCESS_DIAGNOSTIC.search(line) and line not in seen:
            seen.add(line)
            diagnostics.append(i)
    return [*range(min(LIMITS.budget_head_lines, len(units))), *diagnostics, *_tail(len(units)), *_sequence(len(units))]


def _diff_priorities(units: list[str]) -> list[int]:
    headers = [i for i, unit in enumerate(units) if _DIFF_HEADER.match(unit) and not unit.startswith("@@")]
    return [*headers, *_sequence(len(units))]


_PRIORITIES: dict[Category, Callable[[list[str]], list[int]]] = {
    Category.DEFAULT: lambda units: _edges(len(units), LIMITS.budget_head_lines, LIMITS.budget_tail_lines) + _sequence(len(units)),
    Category.FILE: _file_priorities,
    Category.SEARCH: _search_priorities,
    Category.TEST: _test_priorities,
    Category.PROCESS: _process_priorities,
    Category.DIFF: _diff_priorities,
}


def _select(units: list[str], priorities: list[int], budget: Budget) -> list[int]:
    chosen: list[int] = []
    tokens = chars = 0
    marker_tokens, marker_chars = approx_tokens(OMISSION), len(OMISSION)
    for index in priorities:
        if index < 0 or index >= len(units) or (chosen and index in chosen):
            continue
        position = bisect.bisect_left(chosen, index)
        gaps_before = _gaps(chosen, len(units))
        candidate = [*chosen[:position], index, *chosen[position:]]
        gaps_after = _gaps(candidate, len(units))
        next_tokens = tokens + approx_tokens(units[index]) + (gaps_after - gaps_before) * marker_tokens
        next_chars = chars + len(units[index]) + 1 + (gaps_after - gaps_before) * marker_chars
        if not budget.fits(next_tokens, next_chars):
            continue
        chosen, tokens, chars = candidate, next_tokens, next_chars
    return chosen


def _gaps(chosen: list[int], total: int) -> int:
    if not chosen:
        return 0
    gaps = int(chosen[0] > 0) + int(chosen[-1] < total - 1)
    return gaps + sum(1 for a, b in pairwise(chosen) if b - a > 1)


def _render(units: list[str], chosen: list[int]) -> str:
    if not chosen:
        return ""
    out: list[str] = []
    previous = -1
    for index in chosen:
        if index - previous > 1:
            out.append(OMISSION.format(count=index - previous - 1))
        out.append(units[index])
        previous = index
    if previous < len(units) - 1:
        out.append(OMISSION.format(count=len(units) - previous - 1))
    return "\n".join(out)


def _head_tail(text: str, budget: Budget) -> str:
    marker = "\n" + OMISSION.format(count="...") + "\n"
    room = max(0, min(budget.max_chars, budget.max_tokens * LIMITS.chars_per_token) - len(marker))
    head = int(room * LIMITS.budget_head_share)
    return text[:head] + marker + text[len(text) - (room - head):]
