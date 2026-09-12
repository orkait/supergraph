"""WHERE evaluation, column acceleration, and index helpers."""

import re
from functools import lru_cache

import numpy as np

from supergraph.dsl.ast_nodes import (
    AndExpr,
    Condition,
    ContainsCondition,
    DegreeCondition,
    InCondition,
    SimilarCondition,
    LikeCondition,
    NotExpr,
    OrExpr,
)


# LIKE-pattern regex cache. Keyed by the raw pattern string so cache hits
# survive across AST instances (PlanCache may give different call sites
# distinct ast nodes for the same pattern). Pre-fix this memo lived on the
# LikeCondition dataclass itself via ``expr._compiled_re``, which mutated
# the PlanCache-shared AST object — fine single-threaded, a race once the
# plan cache is read concurrently (bug #22). lru_cache is thread-safe in
# CPython.
@lru_cache(maxsize=1024)
def _compile_like_regex(pattern: str):
    parts = []
    for ch in pattern:
        if ch == '%':
            parts.append('.*')
        elif ch == '_':
            parts.append('.')
        else:
            parts.append(re.escape(ch))
    return re.compile(''.join(parts))


class FilteringMixin:

    def _eval_where(self, expr, data: dict) -> bool:
        """Evaluate a WHERE expression against a data dict."""
        if isinstance(expr, Condition):
            return self._eval_condition(expr, data)
        elif isinstance(expr, ContainsCondition):
            actual = data.get(expr.field)
            if actual is None:
                return False
            return expr.value in str(actual)
        elif isinstance(expr, LikeCondition):
            actual = data.get(expr.field)
            if actual is None:
                return False
            # Use the module-level lru_cache instead of stashing the compiled
            # regex on the AST node. Fixes bug #22's thread-safety race by
            # keeping the PlanCache-shared AST immutable.
            regex = _compile_like_regex(expr.pattern)
            return bool(regex.fullmatch(str(actual)))
        elif isinstance(expr, InCondition):
            actual = data.get(expr.field)
            return actual in expr.values
        elif isinstance(expr, SimilarCondition):
            # SIMILAR() requires vector search - cannot evaluate per-node.
            # If we reach here, the vectorized path didn't handle it.
            # Return True to pass through (don't silently reject).
            return True
        elif isinstance(expr, DegreeCondition):
            return self._eval_degree_condition(expr, data)
        elif isinstance(expr, AndExpr):
            return all(self._eval_where(op, data) for op in expr.operands)
        elif isinstance(expr, OrExpr):
            return any(self._eval_where(op, data) for op in expr.operands)
        elif isinstance(expr, NotExpr):
            return not self._eval_where(expr.operand, data)
        return True

    def _eval_condition(self, cond: Condition, data: dict) -> bool:
        """Evaluate a single condition."""
        actual = data.get(cond.field)
        expected = cond.value

        if expected is None:  # NULL check
            if cond.op == "=":
                return actual is None
            elif cond.op == "!=":
                return actual is not None

        if actual is None:
            return False  # NULL doesn't match non-NULL comparisons

        try:
            if cond.op == "=":
                return actual == expected
            elif cond.op == "!=":
                return actual != expected
            elif cond.op == ">":
                return actual > expected
            elif cond.op == "<":
                return actual < expected
            elif cond.op == ">=":
                return actual >= expected
            elif cond.op == "<=":
                return actual <= expected
        except TypeError:
            return False
        return False

    def _eval_degree_condition(self, cond: DegreeCondition, data: dict) -> bool:
        """Evaluate a degree condition. Requires node ID in data dict."""
        node_id = data.get("id")
        if node_id is None:
            return False

        slot = self._resolve_slot(node_id)
        if slot is None:
            return False

        if cond.degree_type == "INDEGREE":
            if cond.edge_kind:
                degree_arr = self.store.edge_matrices.in_degree(cond.edge_kind)
            else:
                # Sum in-degrees across all types
                total = 0
                for etype in self.store.edge_matrices.edge_types:
                    arr = self.store.edge_matrices.in_degree(etype)
                    if arr is not None and slot < len(arr):
                        total += int(arr[slot])
                return self._compare(total, cond.op, cond.value)
        else:  # OUTDEGREE
            if cond.edge_kind:
                degree_arr = self.store.edge_matrices.out_degree(cond.edge_kind)
            else:
                degree_arr = self.store.edge_matrices.out_degree(None)

        if degree_arr is None or slot >= len(degree_arr):
            return self._compare(0, cond.op, cond.value)

        return self._compare(int(degree_arr[slot]), cond.op, cond.value)

    def _compare(self, actual, op, expected) -> bool:
        if op == "=":
            return actual == expected
        if op == "!=":
            return actual != expected
        if op == ">":
            return actual > expected
        if op == "<":
            return actual < expected
        if op == ">=":
            return actual >= expected
        if op == "<=":
            return actual <= expected
        return False

    def _try_index_lookup(self, where, kind_filter: str | None) -> list[dict] | None:
        """Try to use secondary indices for O(1) equality lookups.

        Returns list of matching node dicts if index hit, None otherwise.
        """
        if where is None:
            return None
        expr = where.expr

        # Handle simple: WHERE field = value
        if isinstance(expr, Condition) and expr.op == "=" and expr.field != "kind":
            if expr.field in self.store._indexed_fields:
                slots = self.store.query_by_index(expr.field, expr.value)
                if not slots:
                    return []
                nodes = self.store._materialize_bulk(np.array(slots, dtype=np.int32))
                if kind_filter:
                    nodes = [n for n in nodes if n["kind"] == kind_filter]
                return nodes

        # Handle AND with an indexed equality: WHERE kind = "X" AND name = "Y"
        if isinstance(expr, AndExpr):
            for op in expr.operands:
                if isinstance(op, Condition) and op.op == "=" and op.field != "kind":
                    if op.field in self.store._indexed_fields:
                        slots = self.store.query_by_index(op.field, op.value)
                        remaining_ops = [o for o in expr.operands if o is not op]
                        remaining_ops = [
                            o for o in remaining_ops
                            if not (isinstance(o, Condition) and o.field == "kind" and o.op == "=")
                        ]
                        if not slots:
                            return []
                        nodes = self.store._materialize_bulk(np.array(slots, dtype=np.int32))
                        if kind_filter:
                            nodes = [n for n in nodes if n["kind"] == kind_filter]
                        if remaining_ops:
                            remaining_expr = remaining_ops[0] if len(remaining_ops) == 1 else AndExpr(operands=remaining_ops)
                            nodes = [n for n in nodes if self._eval_where(remaining_expr, n)]
                        return nodes

        return None

    def _extract_kind_from_where(self, where) -> str | None:
        """Extract kind value from WHERE clause, including AND expressions."""
        if where is None:
            return None
        expr = where if not hasattr(where, 'expr') else where.expr
        if isinstance(expr, Condition) and expr.field == "kind" and expr.op == "=":
            return expr.value
        if isinstance(expr, AndExpr):
            for op in expr.operands:
                if isinstance(op, Condition) and op.field == "kind" and op.op == "=":
                    return op.value
        return None

    def _extract_similar_from_where(self, where) -> SimilarCondition | None:
        """Extract SimilarCondition from WHERE clause, including AND expressions."""
        if where is None:
            return None
        expr = where if not hasattr(where, 'expr') else where.expr
        if isinstance(expr, SimilarCondition):
            return expr
        if isinstance(expr, AndExpr):
            for op in expr.operands:
                if isinstance(op, SimilarCondition):
                    return op
        return None

    def _is_simple_kind_filter(self, where) -> bool:
        """Check if WHERE clause is just kind = 'x' with nothing else."""
        if where is None:
            return False
        expr = where.expr
        return isinstance(expr, Condition) and expr.field == "kind" and expr.op == "="

    def _strip_kind_from_expr(self, expr):
        """Remove kind='X' from expression, return remaining or None."""
        if isinstance(expr, Condition) and expr.field == "kind" and expr.op == "=":
            return None
        if isinstance(expr, AndExpr):
            new_ops = [op for op in expr.operands
                       if not (isinstance(op, Condition) and op.field == "kind" and op.op == "=")]
            if not new_ops:
                return None
            if len(new_ops) == 1:
                return new_ops[0]
            return AndExpr(operands=new_ops)
        return expr

    def _contains_degree_condition(self, expr) -> bool:
        """Check if expression tree contains any DegreeCondition."""
        if isinstance(expr, DegreeCondition):
            return True
        if isinstance(expr, AndExpr):
            return any(self._contains_degree_condition(op) for op in expr.operands)
        if isinstance(expr, OrExpr):
            return any(self._contains_degree_condition(op) for op in expr.operands)
        if isinstance(expr, NotExpr):
            return self._contains_degree_condition(expr.operand)
        return False

    def _references_synthetic_fields(self, expr) -> bool:
        """Check if expression references 'kind' or 'id' (stored in separate arrays)."""
        if isinstance(expr, (Condition, ContainsCondition, LikeCondition, InCondition)):
            return expr.field in ("kind", "id")
        if isinstance(expr, AndExpr):
            return any(self._references_synthetic_fields(op) for op in expr.operands)
        if isinstance(expr, OrExpr):
            return any(self._references_synthetic_fields(op) for op in expr.operands)
        if isinstance(expr, NotExpr):
            return self._references_synthetic_fields(expr.operand)
        return False

    def _make_raw_predicate(self, expr):
        """Build a callable(raw_data_dict) -> bool for slot-level filtering.

        Works against materialized data fields (no id/kind).
        Returns None if expression contains DegreeCondition or references
        synthetic fields (kind, id) that aren't in column data.
        """
        if self._contains_degree_condition(expr):
            return None
        if self._references_synthetic_fields(expr):
            return None
        return lambda data, _expr=expr: self._eval_where(_expr, data)

    def _extract_edge_type_from_expr(self, expr) -> str | None:
        """Extract edge type from an arrow expression."""
        if isinstance(expr, Condition) and expr.field == "kind" and expr.op == "=":
            return expr.value
        return None

    def _try_column_filter(self, expr, base_mask: np.ndarray, n: int) -> np.ndarray | None:
        """Try to evaluate expression using column store. Returns bool mask or None."""
        columns = self.store.columns

        if isinstance(expr, Condition):
            if expr.field in ("kind", "id"):
                return None
            mask = columns.get_mask(expr.field, expr.op, expr.value, n)
            if mask is None:
                return None
            return mask & base_mask

        elif isinstance(expr, InCondition):
            if expr.field in ("kind", "id"):
                return None
            mask = columns.get_mask_in(expr.field, expr.values, n)
            if mask is None:
                return None
            return mask & base_mask

        elif isinstance(expr, SimilarCondition):
            if not self._vector_store or not self._embedder:
                return None
            # Vectorize the threshold constraint
            query_vec = self._embedder.encode_queries([expr.query])[0]
            # Use a large k to get all potentially similar nodes, then filter by threshold
            # Since we have an adaptive oversampling loop, we can use it.
            # But for a hard threshold, we want EVERYTHING above score.
            # search() returns (slots, distances)
            # We want dist <= 1.0 - threshold (since distance is 1.0 - sim)
            max_dist = 1.0 - expr.threshold
            # Search with adaptive oversample - let the loop find all matches
            search_k = min(n, int(base_mask.sum()))
            slots, dists = self._vector_store.search(
                query_vec, k=max(search_k, 1), mask=base_mask,
                oversample_factor=getattr(self, '_search_oversample', 10),
            )
            if len(slots) == 0:
                return np.zeros(n, dtype=bool)
            
            mask = np.zeros(n, dtype=bool)
            valid = dists <= max_dist
            mask[slots[valid]] = True
            return mask & base_mask

        elif isinstance(expr, AndExpr):
            result = base_mask.copy()
            for op in expr.operands:
                sub = self._try_column_filter(op, result, n)
                if sub is None:
                    return None
                result = sub
            return result

        elif isinstance(expr, OrExpr):
            result = np.zeros(n, dtype=bool)
            for op in expr.operands:
                sub = self._try_column_filter(op, base_mask, n)
                if sub is None:
                    return None
                result |= sub
            return result

        elif isinstance(expr, NotExpr):
            sub = self._try_column_filter(expr.operand, base_mask, n)
            if sub is None:
                return None
            fields = self._column_fields(expr.operand)
            if fields is None:
                return None
            combined_pres = np.ones(n, dtype=bool)
            for f in fields:
                fp = self.store.columns.get_presence(f, n)
                if fp is None:
                    return None
                combined_pres &= fp
            return ~sub & combined_pres & base_mask

        return None

    def _column_fields(self, expr) -> set[str] | None:
        """Extract field names from expression. None if non-columnarizable."""
        if isinstance(expr, (Condition, ContainsCondition, LikeCondition, InCondition)):
            if expr.field in ("kind", "id"):
                return None
            return {expr.field}
        if isinstance(expr, AndExpr):
            fields = set()
            for op in expr.operands:
                sub = self._column_fields(op)
                if sub is None:
                    return None
                fields |= sub
            return fields
        if isinstance(expr, OrExpr):
            fields = set()
            for op in expr.operands:
                sub = self._column_fields(op)
                if sub is None:
                    return None
                fields |= sub
            return fields
        if isinstance(expr, NotExpr):
            return self._column_fields(expr.operand)
        return None

    def _try_column_nodes(self, expr, kind_filter: str | None) -> list[dict] | None:
        """Try column-accelerated node query. Returns node dicts or None."""
        n = self.store._next_slot
        if n == 0:
            return []
        base_mask = self.store._live_mask(kind_filter)
        col_mask = self._try_column_filter(expr, base_mask, n)
        if col_mask is None:
            return None
        slots = np.nonzero(col_mask)[0]
        return self.store._materialize_bulk(slots)

    def _try_column_count(self, expr, kind_filter: str | None) -> int | None:
        """Try column-accelerated count. Returns count or None."""
        n = self.store._next_slot
        if n == 0:
            return 0
        base_mask = self.store._live_mask(kind_filter)
        col_mask = self._try_column_filter(expr, base_mask, n)
        if col_mask is None:
            return None
        return int(np.count_nonzero(col_mask))

    def _try_column_delete_ids(self, expr, kind_filter: str | None) -> list[str] | None:
        """Try column-accelerated ID query for deletion. Returns IDs or None."""
        n = self.store._next_slot
        if n == 0:
            return []
        base_mask = self.store._live_mask(kind_filter)
        col_mask = self._try_column_filter(expr, base_mask, n)
        if col_mask is None:
            return None
        slots = np.nonzero(col_mask)[0]
        return [
            nid for slot in slots
            if (nid := self.store._slot_to_id(int(slot))) is not None
        ]

    def _try_column_order_by(self, nodes: list[dict], field: str,
                              descending: bool, limit: int | None,
                              offset: int | None) -> list[dict] | None:
        """Column-accelerated ORDER BY - delegates to algos.sort.topk_from_column."""
        from supergraph.algos.sort import topk_from_column

        col_info = self.store.columns.get_column(field, self.store._next_slot)
        if col_info is None:
            return None
        col_data, col_pres, dtype_str = col_info

        slot_to_idx: dict[int, int] = {}
        for i, node in enumerate(nodes):
            slot = self._resolve_slot(node["id"])
            if slot is not None:
                slot_to_idx[slot] = i
        if not slot_to_idx:
            return nodes

        slots = np.fromiter(slot_to_idx.keys(), dtype=np.int32, count=len(slot_to_idx))
        ordered_slots = topk_from_column(
            slots=slots,
            column=col_data,
            presence=col_pres,
            dtype_str=dtype_str,
            descending=descending,
            limit=limit,
            offset=offset,
        )
        if ordered_slots is None:
            return None
        return [nodes[slot_to_idx[int(s)]] for s in ordered_slots]

    def _order_slots_by_column(self, slots: np.ndarray, field: str,
                               descending: bool, limit: int | None,
                               offset: int | None,
                               fallback_predicate=None) -> np.ndarray | None:
        """Sort slot indices by column values - delegates to algos.sort."""
        from supergraph.algos.sort import topk_from_column

        col_info = self.store.columns.get_column(field, self.store._next_slot)
        if col_info is None:
            return None
        col_data, col_pres, dtype_str = col_info

        return topk_from_column(
            slots=slots,
            column=col_data,
            presence=col_pres,
            dtype_str=dtype_str,
            descending=descending,
            limit=limit,
            offset=offset,
            full_sort=fallback_predicate is not None,
        )

    def _materialize_slots_filtered(self, slots: np.ndarray, predicate=None) -> list[dict]:
        """Materialize slot array into node dicts, optionally filtering by predicate."""
        nodes = self.store._materialize_bulk(slots)
        if predicate is None:
            return nodes
        return [node for node in nodes if predicate(node)]
