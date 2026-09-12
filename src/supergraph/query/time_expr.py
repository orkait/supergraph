from __future__ import annotations

from dataclasses import dataclass

from supergraph.query.escape import dsl_time_unit


@dataclass(frozen=True, slots=True)
class TimeExpr:
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
