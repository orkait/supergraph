"""MATCH pattern builder (typed).

Grammar:
  pattern:       match_step (arrow match_step)+
  match_step:    "(" STRING ")"                      -> bound_step
               | "(" IDENTIFIER step_where? ")"      -> var_step
  step_where:    "WHERE" expr
  arrow:         "-[" expr? "]->"

Typed API:

  P.node("fn_main")                                  # ("fn_main")
  P.var("callee")                                    # (callee)
  P.var("callee", where=F.eq("kind", "fn"))          # (callee WHERE kind = "fn")

  pattern = P.node("fn_main").to(P.var("callee"), edge=F.eq("kind", "calls"))
  #   ("fn_main") -[kind = "calls"]-> (callee)

  q.match(pattern, limit=10)
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional

from supergraph.query.escape import dsl_identifier, dsl_literal
from supergraph.query.filters import F, compile_where


@dataclass(frozen=True, slots=True)
class _Step:
    """One node in the pattern. Either bound (quoted id) or var (identifier)."""
    bound_id: Optional[str]      # set -> bound_step; None -> var_step
    var_name: Optional[str]      # set only for var_step
    where: Optional[F]           # step_where, var_step only

    def to_dsl(self) -> str:
        if self.bound_id is not None:
            return f"({dsl_literal(self.bound_id)})"
        assert self.var_name is not None
        dsl_identifier(self.var_name)
        inner = self.var_name
        if self.where is not None:
            w = compile_where(self.where)
            if w:
                inner = f"{self.var_name} WHERE {w}"
        return f"({inner})"


@dataclass(frozen=True, slots=True)
class Pattern:
    """Immutable pattern. Extend via ``.to(step, edge=)``."""
    steps: tuple[_Step, ...]
    edges: tuple[Optional[F], ...]   # one fewer than steps; per-arrow filter

    def to(self, step: "Pattern | _Step", *, edge: F | dict | None = None) -> "Pattern":
        if isinstance(step, Pattern):
            if len(step.steps) != 1:
                raise ValueError("Pattern.to() expects a single-step Pattern or _Step")
            right_step = step.steps[0]
        elif isinstance(step, _Step):
            right_step = step
        else:
            raise TypeError(f"Pattern.to() expects _Step or Pattern, got {type(step).__name__}")
        if isinstance(edge, dict):
            edge = F.from_dict(edge)
        return Pattern(
            steps=self.steps + (right_step,),
            edges=self.edges + (edge,),
        )

    def to_dsl(self) -> str:
        parts = [self.steps[0].to_dsl()]
        for i, step in enumerate(self.steps[1:]):
            edge_f = self.edges[i]
            arrow_inner = ""
            if edge_f is not None:
                w = compile_where(edge_f)
                if w:
                    arrow_inner = w
            parts.append(f"-[{arrow_inner}]->")
            parts.append(step.to_dsl())
        return " ".join(parts)


class _P:
    @staticmethod
    def node(id: str) -> Pattern:
        """``("id")`` - bound step with a specific node id."""
        if not isinstance(id, str) or not id:
            raise ValueError("P.node() requires a non-empty id")
        return Pattern(steps=(_Step(bound_id=id, var_name=None, where=None),), edges=())

    @staticmethod
    def var(name: str, *, where: F | dict | None = None) -> Pattern:
        """``(var)`` - variable step, optionally filtered with WHERE."""
        dsl_identifier(name)
        if isinstance(where, dict):
            where = F.from_dict(where)
        return Pattern(
            steps=(_Step(bound_id=None, var_name=name, where=where),),
            edges=(),
        )


P = _P()
