
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
            regex = _compile_like_regex(expr.pattern)
            return bool(regex.fullmatch(str(actual)))
        elif isinstance(expr, InCondition):
            actual = data.get(expr.field)
            return actual in expr.values
        elif isinstance(expr, SimilarCondition):
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
        actual = data.get(cond.field)
        expected = cond.value

        if expected is None:
            if cond.op == "=":
                return actual is None
            elif cond.op == "!=":
                return actual is not None

        if actual is None:
            return False

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
                total = 0
                for etype in self.store.edge_matrices.edge_types:
                    arr = self.store.edge_matrices.in_degree(etype)
                    if arr is not None and slot < len(arr):
                        total += int(arr[slot])
                return self._compare(total, cond.op, cond.value)
        else:
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
        if where is None:
            return None
        expr = where.expr

        if isinstance(expr, Condition) and expr.op == "=" and expr.field != "kind":
            if expr.field in self.store._indexed_fields:
                slots = self.store.query_by_index(expr.field, expr.value)
                if not slots:
                    return []
                nodes = self.store._materialize_bulk(np.array(slots, dtype=np.int32))
                if kind_filter:
                    nodes = [n for n in nodes if n["kind"] == kind_filter]
                return nodes

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
        if where is None:
            return False
        expr = where.expr
        return isinstance(expr, Condition) and expr.field == "kind" and expr.op == "="

    def _strip_kind_from_expr(self, expr):
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
        if self._contains_degree_condition(expr):
            return None
        if self._references_synthetic_fields(expr):
            return None
        return lambda data, _expr=expr: self._eval_where(_expr, data)

    def _extract_edge_type_from_expr(self, expr) -> str | None:
        if isinstance(expr, Condition) and expr.field == "kind" and expr.op == "=":
            return expr.value
        return None

    def _try_column_filter(self, expr, base_mask: np.ndarray, n: int) -> np.ndarray | None:
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
            query_vec = self._embedder.encode_queries([expr.query])[0]
            max_dist = 1.0 - expr.threshold
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
        n = self.store._next_slot
        if n == 0:
            return 0
        base_mask = self.store._live_mask(kind_filter)
        col_mask = self._try_column_filter(expr, base_mask, n)
        if col_mask is None:
            return None
        return int(np.count_nonzero(col_mask))

    def _try_column_delete_ids(self, expr, kind_filter: str | None) -> list[str] | None:
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
        nodes = self.store._materialize_bulk(slots)
        if predicate is None:
            return nodes
        return [node for node in nodes if predicate(node)]
