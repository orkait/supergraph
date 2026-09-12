"""Typed aggregate functions for AGGREGATE NODES SELECT + HAVING.

Grammar:
  agg_func:     "COUNT" "(" ")"                    -> agg_count
             | "COUNT" "DISTINCT" "(" IDENT ")"     -> agg_count_distinct
             | "SUM" "(" IDENT ")"                  -> agg_sum
             | "AVG" "(" IDENT ")"                  -> agg_avg
             | "MIN" "(" IDENT ")"                  -> agg_min
             | "MAX" "(" IDENT ")"                  -> agg_max
  having_expr:  agg_func OP value

Typed API:

  from supergraph.query import agg

  agg.count()                     -> COUNT()
  agg.count_distinct("topic")     -> COUNT DISTINCT(topic)
  agg.sum("importance")           -> SUM(importance)
  agg.avg("importance")
  agg.min("x") / agg.max("x")

Comparison operators build a HavingExpr:

  agg.avg("importance") > 0.5     -> AVG(importance) > 0.5
  agg.count() >= 10

  q.aggregate_nodes(select=[agg.count(), agg.avg("importance")],
                    having=agg.avg("importance") > 0.5)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from supergraph.query.escape import dsl_identifier, dsl_literal


@dataclass(frozen=True, slots=True)
class AggFunc:
    """Typed agg_func. Used both inside SELECT lists and HAVING expressions."""
    kind: str           # "count" | "count_distinct" | "sum" | "avg" | "min" | "max"
    field: str | None   # None only for count()

    def to_dsl(self) -> str:
        if self.kind == "count":
            return "COUNT()"
        if self.kind == "count_distinct":
            return f"COUNT DISTINCT({dsl_identifier(self.field)})"
        keyword = {"sum": "SUM", "avg": "AVG", "min": "MIN", "max": "MAX"}[self.kind]
        return f"{keyword}({dsl_identifier(self.field)})"

    def __str__(self) -> str:
        return self.to_dsl()

    # -- HAVING expression builders via comparison operators --

    def _having(self, op: str, value: Any) -> "HavingExpr":
        return HavingExpr(self, op, value)

    def __gt__(self, v: Any)  -> "HavingExpr": return self._having(">",  v)
    def __ge__(self, v: Any)  -> "HavingExpr": return self._having(">=", v)
    def __lt__(self, v: Any)  -> "HavingExpr": return self._having("<",  v)
    def __le__(self, v: Any)  -> "HavingExpr": return self._having("<=", v)
    def __eq__(self, v: Any)  -> "HavingExpr": return self._having("=",  v)    # type: ignore[override]
    def __ne__(self, v: Any)  -> "HavingExpr": return self._having("!=", v)    # type: ignore[override]
    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class HavingExpr:
    func: AggFunc
    op: str
    value: Any

    def to_dsl(self) -> str:
        return f"{self.func.to_dsl()} {self.op} {dsl_literal(self.value)}"


class _Agg:
    @staticmethod
    def count() -> AggFunc:
        return AggFunc("count", None)

    @staticmethod
    def count_distinct(field: str) -> AggFunc:
        dsl_identifier(field)
        return AggFunc("count_distinct", field)

    @staticmethod
    def sum(field: str) -> AggFunc:
        dsl_identifier(field)
        return AggFunc("sum", field)

    @staticmethod
    def avg(field: str) -> AggFunc:
        dsl_identifier(field)
        return AggFunc("avg", field)

    @staticmethod
    def min(field: str) -> AggFunc:
        dsl_identifier(field)
        return AggFunc("min", field)

    @staticmethod
    def max(field: str) -> AggFunc:
        dsl_identifier(field)
        return AggFunc("max", field)


agg = _Agg()
