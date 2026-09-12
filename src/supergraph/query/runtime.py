"""Query builder runtime: the ``Query`` class + compose + execute.

Design: ``Query`` stores a ``_verb`` tag and a ``_params`` dict. The
``.dsl()`` method dispatches to a per-verb compiler that knows the
grammar-correct clause order (different per verb - REMEMBER wants
AT/TOKENS/LIMIT/WHERE, NODES wants WHERE/ORDER/LIMIT, etc.).

Modifiers return a new ``Query`` with updated params; input is never
mutated.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

from supergraph.query.filters import F, compile_where

if TYPE_CHECKING:
    from supergraph.store import SuperGraph


QueryKind = Literal["read", "write", "sys", "vault", "batch", "raw", "control"]


# Per-verb DSL compiler registry. Populated by verb modules at import time.
_COMPILERS: dict[str, Callable[[dict], str]] = {}


def register_compiler(verb: str, fn: Callable[[dict], str]) -> None:
    """Verb modules register their compiler so ``Query.dsl()`` can dispatch."""
    _COMPILERS[verb] = fn


# Modifier -> read verbs that accept it. Others raise on mutate.
_READ_MODIFIERS: set[str] = {"where", "limit", "tokens", "at", "order_by"}


@dataclass(frozen=True, slots=True)
class Query:
    """Immutable DSL query. Compile via ``.dsl()``, run via ``.execute(gs)``."""

    _verb: str
    _params: dict
    _kind: QueryKind = "read"

    # -- Terminal operations -------------------------------------------

    def dsl(self) -> str:
        compiler = _COMPILERS.get(self._verb)
        if compiler is None:
            raise RuntimeError(
                f"no compiler for verb {self._verb!r}. "
                f"registered: {sorted(_COMPILERS)}"
            )
        return compiler(self._params)

    def execute(self, gs: "SuperGraph") -> Any:
        if gs is None:
            raise TypeError("Query.execute() requires a SuperGraph instance, got None")
        return gs.execute(self.dsl())

    # -- Modifiers (read only) -----------------------------------------

    def _require_read(self, op: str) -> None:
        if self._kind != "read":
            raise TypeError(
                f"Query.{op}() is only valid on read queries; "
                f"this is a {self._kind!r} query"
            )

    def _with_param(self, key: str, value: Any) -> "Query":
        new_params = {**self._params, key: value}
        return replace(self, _params=new_params)

    def limit(self, n: int) -> "Query":
        self._require_read("limit")
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError(f"limit must be a non-negative int, got {n!r}")
        return self._with_param("limit", n)

    def tokens(self, n: int) -> "Query":
        self._require_read("tokens")
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError(f"tokens must be a non-negative int, got {n!r}")
        return self._with_param("tokens", n)

    def at(self, when: str) -> "Query":
        self._require_read("at")
        if not isinstance(when, str) or not when:
            raise ValueError(f"at must be a non-empty str, got {when!r}")
        return self._with_param("at", when)

    def order_by(self, expr: str) -> "Query":
        self._require_read("order_by")
        if not isinstance(expr, str) or not expr:
            raise ValueError(f"order_by must be a non-empty str, got {expr!r}")
        return self._with_param("order_by", expr)

    def where(self, predicate: F | dict | None) -> "Query":
        """AND-combine a predicate with the existing WHERE."""
        self._require_read("where")
        if predicate is None:
            return self
        if isinstance(predicate, dict):
            predicate = F.from_dict(predicate)
        if not isinstance(predicate, F):
            raise TypeError(f"where predicate must be F or dict, got {type(predicate).__name__}")
        existing = self._params.get("where")
        combined = predicate if existing is None else (existing & predicate)
        return self._with_param("where", combined)

    def with_(self, **kw: Any) -> "Query":
        """Replace any named read-modifier. Unknown kwargs raise."""
        self._require_read("with_")
        unknown = set(kw) - _READ_MODIFIERS
        if unknown:
            raise TypeError(
                f"with_() got unexpected kwargs: {sorted(unknown)}; "
                f"allowed: {sorted(_READ_MODIFIERS)}"
            )
        out = self
        for key, val in kw.items():
            if val is None:
                p = dict(out._params)
                p.pop(key, None)
                out = replace(out, _params=p)
            elif key == "where":
                out = out.where(val)
            elif key == "limit":
                out = out.limit(val)
            elif key == "tokens":
                out = out.tokens(val)
            elif key == "at":
                out = out.at(val)
            elif key == "order_by":
                out = out.order_by(val)
        return out

    # -- Functional composition ----------------------------------------

    def pipe(self, fn: Callable[..., "Query"], *args: Any, **kwargs: Any) -> "Query":
        out = fn(self, *args, **kwargs)
        if not isinstance(out, Query):
            raise TypeError(f"pipe() function must return a Query, got {type(out).__name__}")
        return out

    # -- Batch compose -------------------------------------------------

    def __or__(self, other: "Query") -> "Query":
        if not isinstance(other, Query):
            return NotImplemented
        left = [self.dsl()] if self._kind != "batch" else [self._params["text"]]
        right = [other.dsl()] if other._kind != "batch" else [other._params["text"]]
        combined = "\n".join(left + right)
        return Query(_verb="batch", _params={"text": combined}, _kind="batch")

    def __repr__(self) -> str:
        try:
            text = self.dsl()
        except Exception:
            text = f"<compile-error verb={self._verb}>"
        if len(text) > 80:
            text = text[:77] + "..."
        return f"<Query[{self._kind}]: {text}>"

    def __str__(self) -> str:
        return self.dsl()


# -- Batch compiler (always available) -------------------------------------

def _compile_batch(params: dict) -> str:
    return params["text"]


def _compile_raw(params: dict) -> str:
    return params["text"]


register_compiler("batch", _compile_batch)
register_compiler("raw", _compile_raw)
