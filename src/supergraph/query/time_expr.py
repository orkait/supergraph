"""Time expression helpers.

Grammar:
  time_expr: "NOW" "(" ")" "-" NUMBER TIME_UNIT  -> time_offset
           | "NOW" "(" ")"                       -> time_now
           | "TODAY"                             -> time_today
           | "YESTERDAY"                         -> time_yesterday
  TIME_UNIT: /[smhd]/

These appear as values inside WHERE expressions (e.g. ``kind = "m" AND
__event_at__ > NOW() - 7d``) and in ASSERT's EVENT_AT / CREATE NODE's
EVENT_AT slots.
"""
from __future__ import annotations

from dataclasses import dataclass

from supergraph.query.escape import dsl_time_unit


@dataclass(frozen=True, slots=True)
class TimeExpr:
    """Symbolic time. Compiles to a DSL value via ``.to_dsl()``."""
    text: str

    def to_dsl(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"TimeExpr({self.text!r})"


class _Time:
    @staticmethod
    def now() -> TimeExpr:
        return TimeExpr("NOW()")

    @staticmethod
    def today() -> TimeExpr:
        return TimeExpr("TODAY")

    @staticmethod
    def yesterday() -> TimeExpr:
        return TimeExpr("YESTERDAY")

    @staticmethod
    def now_minus(n: int, unit: str) -> TimeExpr:
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError(f"now_minus n must be a non-negative int, got {n!r}")
        dsl_time_unit(unit)
        return TimeExpr(f"NOW() - {n}{unit}")


Time = _Time()
