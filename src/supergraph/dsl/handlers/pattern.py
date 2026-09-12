
import logging

import numpy as np

from supergraph.dsl.handlers._registry import handles
from supergraph.dsl.ast_nodes import MatchQuery, MatchPattern
from supergraph.core.types import Result
from supergraph.core.errors import CostThresholdExceeded
from supergraph.dsl.cost_estimator import estimate_match_cost

logger = logging.getLogger(__name__)

_DEFAULT_MATCH_FRONTIER_CAP = 10_000


class PatternHandlers:

    @handles(MatchQuery)
    def _match(self, q: MatchQuery) -> Result:
        pattern = q.pattern

        cost = estimate_match_cost(pattern, self.store.edge_matrices, threshold=self.cost_threshold)
        if cost.rejected:
            raise CostThresholdExceeded(cost.estimated_frontier, self.cost_threshold)

        frontier_cap = (
            max(q.limit.value, _DEFAULT_MATCH_FRONTIER_CAP)
            if q.limit is not None
            else _DEFAULT_MATCH_FRONTIER_CAP
        )
        bindings, edges = self._execute_match_pattern(
            pattern, frontier_cap=frontier_cap,
        )

        bindings = [
            b for b in bindings
            if all(self._is_visible_by_id(nid) for nid in b.values())
        ]
        edges = [
            e for e in edges
            if self._is_visible_by_id(e["source"]) and self._is_visible_by_id(e["target"])
        ]

        if q.limit:
            bindings = bindings[: q.limit.value]

        return Result(
            kind="match",
            data={"bindings": bindings, "edges": edges},
            count=len(bindings),
        )

    def _execute_match_pattern(
        self,
        pattern: MatchPattern,
        frontier_cap: int = _DEFAULT_MATCH_FRONTIER_CAP,
    ) -> tuple[list[dict], list[dict]]:
        steps = pattern.steps
        arrows = pattern.arrows

        first_step = steps[0]
        if first_step.bound_id:
            start_slot = self._resolve_slot(first_step.bound_id)
            if start_slot is None:
                return [], []
            current_slots = [start_slot]
        else:
            n_total = self.store._next_slot
            mask = self._compute_live_mask(n_total)
            kind = self._extract_kind_from_where(first_step.where)
            if kind:
                mask &= self.store._live_mask(kind)
            
            remaining = None
            if first_step.where:
                remaining = self._strip_kind_from_expr(first_step.where)
                if remaining:
                    col_mask = self._try_column_filter(remaining, mask, n_total)
                    if col_mask is not None:
                        mask = col_mask
                        remaining = None
            
            current_slots = np.where(mask)[0].tolist()
            if remaining:
                all_nodes = self.store._materialize_bulk(np.array(current_slots, dtype=np.int32))
                filtered_slots = []
                for i, node in enumerate(all_nodes):
                    if self._eval_where(remaining, node):
                        filtered_slots.append(current_slots[i])
                current_slots = filtered_slots

        if not current_slots:
            return [], []

        paths = [[] for _ in current_slots]
        edge_trails = [[] for _ in current_slots]

        for i, slot in enumerate(current_slots):
            if first_step.variable:
                nid = self.store._slot_to_id(slot)
                paths[i].append((first_step.variable, nid))
            elif first_step.bound_id:
                paths[i].append(("_start", first_step.bound_id))

        for arrow, next_step in zip(arrows, steps[1:]):
            edge_type = self._extract_edge_type_from_expr(arrow.expr)
            new_paths = []
            new_slots = []
            new_edge_trails = []

            for i, slot in enumerate(current_slots):
                source_nid = self.store._slot_to_id(slot)
                neighbors = self.store.edge_matrices.neighbors_out(slot, edge_type)

                for nb in neighbors:
                    nb = int(nb)
                    nid = self.store._slot_to_id(nb)
                    if nid is None:
                        continue

                    if next_step.bound_id:
                        if nid != next_step.bound_id:
                            continue

                    if next_step.where:
                        node_data = self.store.get_node(nid)
                        if not node_data or not self._eval_where(
                            next_step.where, node_data
                        ):
                            continue

                    new_path = list(paths[i])
                    if next_step.variable:
                        new_path.append((next_step.variable, nid))
                    elif next_step.bound_id:
                        new_path.append(("_bound", nid))

                    new_edge_trail = list(edge_trails[i]) + [
                        {"source": source_nid, "target": nid, "kind": edge_type or ""}
                    ]

                    new_paths.append(new_path)
                    new_slots.append(nb)
                    new_edge_trails.append(new_edge_trail)

            current_slots = new_slots
            paths = new_paths
            edge_trails = new_edge_trails

            if not current_slots:
                return [], []

            if len(current_slots) > frontier_cap:
                logger.warning(
                    "MATCH expansion hit frontier cap %d (have %d bindings) — "
                    "truncating. Increase LIMIT or cost_threshold to see more.",
                    frontier_cap, len(current_slots),
                )
                current_slots = current_slots[:frontier_cap]
                paths = paths[:frontier_cap]
                edge_trails = edge_trails[:frontier_cap]

        bindings = []
        for path in paths:
            binding = {}
            for var_name, node_id in path:
                if not var_name.startswith("_"):
                    binding[var_name] = node_id
            bindings.append(binding)

        seen_edges: dict[str, dict] = {}
        for trail in edge_trails:
            for e in trail:
                key = f"{e['source']}->{e['target']}:{e['kind']}"
                seen_edges[key] = e

        return bindings, list(seen_edges.values())
