"""Typed EVOLVE rule when/then constructors.

Grammar:
  evolve_when_clause:  "WHEN" evolve_condition ("AND" evolve_condition)*
  evolve_condition:    IDENTIFIER EVOLVE_OP NUMBER
  EVOLVE_OP:           ">=" | "<=" | "==" | "!=" | ">" | "<"

  evolve_then_clause:  "THEN" evolve_action
  evolve_action:       "SET" IDENT "=" evolve_value                    -> evolve_action_set
                     | "ADJUST" IDENT "BY" NUMBER "UNTIL" NUMBER        -> evolve_action_adjust_until
                     | "ADJUST" IDENT "BY" NUMBER                       -> evolve_action_adjust
                     | "ADD" IDENT STRING                               -> evolve_action_add
                     | "REMOVE" IDENT STRING                            -> evolve_action_remove
                     | "RUN" IDENT+                                     -> evolve_action_run
  evolve_value:        "[" NUMBER ("," NUMBER)* "]"                     -> evolve_value_list
                     | NUMBER                                           -> evolve_value_scalar

Typed API:

  from supergraph.query import EvolveWhen as W, EvolveThen as A

  q.sys.evolve.rule(
      "r1",
      when=[W.cond("recall_hit_rate", "<=", 0.4)],
      then=[A.run("SYS", "REEMBED")],
      cooldown=86400,
  )

Strings still accepted for backwards-compat; typed objects are the
recommended path forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from supergraph.query.escape import dsl_identifier, dsl_literal


_VALID_OPS = {">=", "<=", "==", "!=", ">", "<"}


@dataclass(frozen=True, slots=True)
class EvolveCondition:
    signal: str
    op: str
    value: int | float

    def to_dsl(self) -> str:
        if self.op not in _VALID_OPS:
            raise ValueError(f"EVOLVE_OP must be in {sorted(_VALID_OPS)}, got {self.op!r}")
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
            raise TypeError(f"EVOLVE condition value must be a number, got {self.value!r}")
        return f"{dsl_identifier(self.signal)} {self.op} {self.value}"


@dataclass(frozen=True, slots=True)
class EvolveAction:
    kind: str          # "set" | "adjust" | "adjust_until" | "add" | "remove" | "run"
    target: str | None
    value: Any         # meaning depends on kind

    def to_dsl(self) -> str:
        if self.kind == "set":
            return f"SET {dsl_identifier(self.target)} = {_render_evolve_value(self.value)}"
        if self.kind == "adjust":
            return f"ADJUST {dsl_identifier(self.target)} BY {self.value}"
        if self.kind == "adjust_until":
            by, until = self.value
            return f"ADJUST {dsl_identifier(self.target)} BY {by} UNTIL {until}"
        if self.kind == "add":
            return f"ADD {dsl_identifier(self.target)} {dsl_literal(self.value)}"
        if self.kind == "remove":
            return f"REMOVE {dsl_identifier(self.target)} {dsl_literal(self.value)}"
        if self.kind == "run":
            # value is a tuple of IDENT tokens
            if not self.value:
                raise ValueError("EvolveAction.run requires at least one identifier")
            return "RUN " + " ".join(dsl_identifier(i) for i in self.value)
        raise ValueError(f"unknown EvolveAction kind {self.kind!r}")


def _render_evolve_value(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        if not v:
            raise ValueError("evolve SET value list must be non-empty")
        for x in v:
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                raise TypeError("evolve SET list items must be numbers")
        return "[" + ", ".join(str(x) for x in v) + "]"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    raise TypeError(f"evolve SET value must be a number or list of numbers, got {v!r}")


class _EvolveWhen:
    @staticmethod
    def cond(signal: str, op: str, value: int | float) -> EvolveCondition:
        return EvolveCondition(signal, op, value)


class _EvolveThen:
    @staticmethod
    def set(target: str, value: int | float | list | tuple) -> EvolveAction:
        return EvolveAction("set", target, value)

    @staticmethod
    def adjust(target: str, by: int | float) -> EvolveAction:
        return EvolveAction("adjust", target, by)

    @staticmethod
    def adjust_until(target: str, by: int | float, until: int | float) -> EvolveAction:
        return EvolveAction("adjust_until", target, (by, until))

    @staticmethod
    def add(target: str, value: str) -> EvolveAction:
        return EvolveAction("add", target, value)

    @staticmethod
    def remove(target: str, value: str) -> EvolveAction:
        return EvolveAction("remove", target, value)

    @staticmethod
    def run(*tokens: str) -> EvolveAction:
        if not tokens:
            raise ValueError("EvolveThen.run requires at least one identifier")
        return EvolveAction("run", None, tuple(tokens))


EvolveWhen = _EvolveWhen()
EvolveThen = _EvolveThen()
