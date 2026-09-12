"""Node CRUD + COUNT handlers."""

import numpy as np

from supergraph.dsl.handlers._registry import handles
from supergraph.dsl.ast_nodes import (
    CountQuery, NodeQuery, NodesQuery,
)
from supergraph.core.types import Result


class NodeHandlers:

    @handles(NodeQuery)
    def _node(self, q: NodeQuery) -> Result:
        data = self.store.get_node(q.id)
        if data is not None:
            slot = self._resolve_slot(q.id)
            if slot is not None:
                if not self._is_slot_visible(slot):
                    data = None
                elif q.with_document and self._document_store:
                    doc = self._document_store.get_document(slot)
                    if doc:
                        content, ctype = doc
                        if ctype.startswith("text"):
                            data["_document"] = content.decode("utf-8", errors="replace")
                        else:
                            data["_document"] = content
                        data["_document_type"] = ctype
        return Result(kind="node", data=data, count=1 if data else 0)

    @handles(NodesQuery)
    def _nodes(self, q: NodesQuery) -> Result:
        kind_filter = self._extract_kind_from_where(q.where) if q.where else None

        nodes = self._try_index_lookup(q.where, kind_filter) if q.where else None

        if nodes is not None:
            nodes = self._filter_visible(nodes)
            if q.order:
                reverse = q.order.direction == "DESC"
                col_sorted = self._try_column_order_by(
                    nodes, q.order.field, reverse,
                    q.limit.value if q.limit else None,
                    q.offset.value if q.offset else None,
                )
                if col_sorted is not None:
                    nodes = col_sorted
                else:
                    nodes.sort(
                        key=lambda n: (n.get(q.order.field) is None, n.get(q.order.field, "")),
                        reverse=reverse,
                    )
                    if q.offset:
                        nodes = nodes[q.offset.value:]
                    if q.limit:
                        nodes = nodes[:q.limit.value]
            else:
                if q.offset:
                    nodes = nodes[q.offset.value:]
                if q.limit:
                    nodes = nodes[:q.limit.value]
            return Result(kind="nodes", data=nodes, count=len(nodes))

        n = self.store._next_slot
        if n == 0:
            return Result(kind="nodes", data=[], count=0)

        final_mask = self._compute_live_mask(n)

        if kind_filter:
            kind_mask = self.store._live_mask(kind_filter)
            final_mask = final_mask & kind_mask

        fallback_predicate = None
        if q.where and not self._is_simple_kind_filter(q.where):
            remaining = self._strip_kind_from_expr(q.where.expr)
            if remaining is not None:
                col_mask = self._try_column_filter(remaining, final_mask, n)
                if col_mask is not None:
                    final_mask = col_mask
                else:
                    raw_pred = self._make_raw_predicate(remaining)
                    if raw_pred is not None:
                        fallback_predicate = lambda node, _expr=remaining: self._eval_where(_expr, node)
                    else:
                        fallback_predicate = lambda node, _expr=q.where.expr: self._eval_where(_expr, node)

        slots = np.where(final_mask)[0]

        if q.order:
            reverse = q.order.direction == "DESC"
            col_sorted_slots = self._order_slots_by_column(
                slots, q.order.field, reverse,
                q.limit.value if q.limit else None,
                q.offset.value if q.offset else None,
                fallback_predicate,
            )
            if col_sorted_slots is not None:
                # When the predicate fallback is in play, `_order_slots_by_column`
                # asked `topk_slot_order` for a FULL sort (full_sort=True) and
                # returned the whole sorted slot array, unsliced. The caller
                # must apply offset/limit AFTER the Python-side predicate filter
                # so that "NODES WHERE x CONTAINS y ORDER BY z LIMIT N" returns
                # at most N rows. Without this slicing the LIMIT clause was
                # silently ignored under any non-column-filter WHERE — bug #90.
                nodes = self.store._materialize_bulk(col_sorted_slots)
                if fallback_predicate:
                    nodes = [n for n in nodes if fallback_predicate(n)]
                    if q.offset:
                        nodes = nodes[q.offset.value:]
                    if q.limit:
                        nodes = nodes[:q.limit.value]
                return Result(kind="nodes", data=nodes, count=len(nodes))
            else:
                nodes = self._materialize_slots_filtered(slots, fallback_predicate)
                nodes.sort(
                    key=lambda nd: (nd.get(q.order.field) is None, nd.get(q.order.field, "")),
                    reverse=reverse,
                )
                if q.offset:
                    nodes = nodes[q.offset.value:]
                if q.limit:
                    nodes = nodes[:q.limit.value]
                return Result(kind="nodes", data=nodes, count=len(nodes))

        if fallback_predicate is None:
            if q.offset:
                slots = slots[q.offset.value:]
            if q.limit:
                slots = slots[:q.limit.value]
            result = self.store._materialize_bulk(slots)
        else:
            result = self._materialize_slots_filtered(slots, fallback_predicate)
            if q.offset:
                result = result[q.offset.value:]
            if q.limit:
                result = result[:q.limit.value]

        return Result(kind="nodes", data=result, count=len(result))

    @handles(CountQuery)
    def _count(self, q: CountQuery) -> Result:
        if q.target == "NODES":
            if q.where:
                kind_filter = self._extract_kind_from_where(q.where)
                remaining = self._strip_kind_from_expr(q.where.expr)
                if remaining is None:
                    count = self.store.count_nodes(kind=kind_filter)
                else:
                    col_count = self._try_column_count(remaining, kind_filter)
                    if col_count is not None:
                        count = col_count
                    else:
                        raw_pred = self._make_raw_predicate(remaining)
                        if raw_pred is not None:
                            count = self.store.count_nodes(kind=kind_filter, predicate=raw_pred)
                        else:
                            nodes = self.store.get_all_nodes(kind=kind_filter)
                            count = sum(1 for n in nodes if self._eval_where(q.where.expr, n))
            else:
                # Honor namespace/context isolation: a raw node_count would
                # leak namespaced (and context) nodes into the default view.
                if (getattr(self.store, "_active_namespace", None) is not None
                        or self.store.columns.has_column("__namespace__")
                        or self.store._active_context is not None):
                    n = self.store._next_slot
                    count = int(self._compute_live_mask(n).sum()) if n else 0
                else:
                    count = self.store.node_count
        else:
            if q.where:
                kind_filter = self._extract_kind_from_where(q.where)
                if self._is_simple_kind_filter(q.where) and kind_filter:
                    count = len(self.store._edges_by_type.get(kind_filter, []))
                else:
                    edges = self.store.get_all_edges()
                    count = sum(
                        1 for e in edges
                        if self._eval_where(q.where.expr, e) and self._edge_visible(e)
                    )
            else:
                # Honor namespace/context isolation: an edge is visible only when
                # BOTH endpoints are visible under the current view, else a raw
                # edge_count leaks edges between namespaced (or context) nodes.
                if (getattr(self.store, "_active_namespace", None) is not None
                        or self.store.columns.has_column("__namespace__")
                        or self.store._active_context is not None):
                    n = self.store._next_slot
                    live = self._compute_live_mask(n)
                    count = sum(
                        1
                        for edges in self.store._edges_by_type.values()
                        for (s, t, _d) in edges
                        if s < n and t < n and live[s] and live[t]
                    )
                else:
                    count = self.store.edge_count
        return Result(kind="count", data=count, count=count)

    def _edge_visible(self, edge: dict) -> bool:
        """True when both edge endpoints are visible under the current view
        (namespace/context/TTL/retraction). Cheap no-op when nothing is scoped."""
        if (getattr(self.store, "_active_namespace", None) is None
                and not self.store.columns.has_column("__namespace__")
                and self.store._active_context is None):
            return True
        return self._is_visible_by_id(edge["source"]) and self._is_visible_by_id(edge["target"])
