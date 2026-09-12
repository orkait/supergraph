"""Graph traversal handlers: TRAVERSE, PATH, SHORTEST PATH, ANCESTORS, DESCENDANTS, SUBGRAPH, COMMON NEIGHBORS, DISTANCE."""

import numpy as np

from supergraph.dsl.handlers._registry import handles
from supergraph.dsl.ast_nodes import (
    AncestorsQuery, CommonNeighborsQuery, DescendantsQuery, DistanceQuery,
    PathQuery, PathsQuery, ShortestPathQuery, SubgraphQuery, TraverseQuery,
    WeightedDistanceQuery, WeightedShortestPathQuery,
)
from supergraph.core.path import (
    bfs_traverse, bidirectional_bfs, common_neighbors, dijkstra, find_all_paths,
)
from supergraph.core.types import Result
from supergraph.core.errors import CostThresholdExceeded
from supergraph.dsl.cost_estimator import estimate_traverse_cost


class TraversalHandlers:

    def _visibility_masked_matrix(self, matrix):
        """Return a copy of ``matrix`` with rows+columns for invisible slots zeroed.

        Pre-fix, PATH/SHORTEST PATH queries did BFS over the full matrix and
        then discarded any returned path that touched an invisible node,
        with no fallback to try an alternate route (bugs #39/#44). By
        pre-masking the matrix, invisible nodes become unreachable from
        the BFS's point of view — the algorithm naturally routes around
        them and finds the shortest path among visible nodes.

        Returns None if the matrix is None or fully empty.
        """
        if matrix is None:
            return None
        n = matrix.shape[0]
        if n == 0:
            return matrix
        live_mask = self._compute_live_mask(n)
        if live_mask.all():
            # Fast path: no hidden nodes, original matrix is fine.
            return matrix
        # Build a diagonal selector: rows * diag * cols effectively keeps
        # an entry only if BOTH its source and target are visible. For a
        # CSR, the cheapest way is to element-multiply a mask column
        # after a row-slice. scipy.sparse handles the broadcasting.
        from scipy.sparse import diags
        d = diags(live_mask.astype(np.float32))
        # d @ M @ d: rows filtered by visibility then cols filtered. Cast
        # back to CSR since BFS expects the indices/indptr layout.
        masked = (d @ matrix @ d).tocsr()
        return masked

    @handles(TraverseQuery)
    def _traverse(self, q: TraverseQuery) -> Result:
        slot = self._resolve_slot(q.start_id)
        if slot is None:
            return Result(kind="nodes", data=[], count=0)
        edge_type = self._extract_kind_from_where(q.where)
        matrix = self.store.edge_matrices.get({edge_type} if edge_type else None)
        if matrix is None:
            node_data = self.store.get_node(q.start_id)
            return Result(kind="nodes", data=[node_data] if node_data else [], count=1 if node_data else 0)
        cost = estimate_traverse_cost(q.depth, self.store.edge_matrices, edge_type, threshold=self.cost_threshold)
        if cost.rejected:
            raise CostThresholdExceeded(cost.estimated_frontier, self.cost_threshold)
        visited = bfs_traverse(matrix, slot, q.depth)
        nodes = []
        for node_idx, depth in visited:
            nid = self.store._slot_to_id(node_idx)
            if nid:
                node = self.store.get_node(nid)
                if node:
                    node["_depth"] = depth
                    nodes.append(node)
        nodes = self._filter_visible(nodes)
        if q.limit:
            nodes = nodes[:q.limit.value]
        return Result(kind="nodes", data=nodes, count=len(nodes))

    @handles(SubgraphQuery)
    def _subgraph(self, q: SubgraphQuery) -> Result:
        slot = self._resolve_slot(q.start_id)
        if slot is None:
            return Result(kind="subgraph", data={"nodes": [], "edges": []}, count=0)
        matrix = self.store.edge_matrices.get(None)
        if matrix is None:
            node_data = self.store.get_node(q.start_id)
            return Result(kind="subgraph", data={"nodes": [node_data] if node_data else [], "edges": []}, count=1 if node_data else 0)
        visited = bfs_traverse(matrix, slot, q.depth)
        visited_slots = {node_idx for node_idx, _ in visited}
        nodes = []
        for node_idx, depth in visited:
            nid = self.store._slot_to_id(node_idx)
            if nid:
                node = self.store.get_node(nid)
                if node:
                    nodes.append(node)
        nodes = self._filter_visible(nodes)
        visible_ids = {n["id"] for n in nodes}
        edges = []
        for node_idx in visited_slots:
            nid = self.store._slot_to_id(node_idx)
            if nid and nid in visible_ids:
                for e in self.store.get_edges_from(nid):
                    if e["target"] in visible_ids:
                        edges.append(e)
        return Result(kind="subgraph", data={"nodes": nodes, "edges": edges}, count=len(nodes))

    @handles(PathQuery)
    def _path(self, q: PathQuery) -> Result:
        src_slot = self._resolve_slot(q.from_id)
        tgt_slot = self._resolve_slot(q.to_id)
        if src_slot is None or tgt_slot is None:
            return Result(kind="path", data=None, count=0)
        edge_type = self._extract_kind_from_where(q.where)
        matrix = self.store.edge_matrices.get({edge_type} if edge_type else None)
        matrix = self._visibility_masked_matrix(matrix)
        if matrix is None:
            return Result(kind="path", data=None, count=0)
        matrix_t = matrix.T.tocsr()
        path = bidirectional_bfs(matrix, matrix_t, src_slot, tgt_slot, q.max_depth)
        if path is None:
            return Result(kind="path", data=None, count=0)
        path_ids = [self.store._slot_to_id(s) for s in path]
        # Pre-masked: if BFS returned a path, every hop is visible. But we
        # still guard against _slot_to_id returning None for a tombstoned
        # slot that appeared visible at mask-time but got deleted before
        # materialization.
        if any(pid is None for pid in path_ids):
            return Result(kind="path", data=None, count=0)
        return Result(kind="path", data=path_ids, count=len(path_ids))

    @handles(PathsQuery)
    def _paths(self, q: PathsQuery) -> Result:
        src_slot = self._resolve_slot(q.from_id)
        tgt_slot = self._resolve_slot(q.to_id)
        if src_slot is None or tgt_slot is None:
            return Result(kind="paths", data=[], count=0)
        edge_type = self._extract_kind_from_where(q.where)
        matrix = self.store.edge_matrices.get({edge_type} if edge_type else None)
        # Visibility pre-filter: invisible nodes can't participate in any
        # returned path since their edges are masked to zero. Pre-fix, the
        # all-paths enumeration returned paths through invisible nodes
        # and then discarded them post-hoc (bug #44).
        matrix = self._visibility_masked_matrix(matrix)
        if matrix is None:
            return Result(kind="paths", data=[], count=0)
        paths = find_all_paths(matrix, src_slot, tgt_slot, q.max_depth)
        result = []
        for path in paths:
            path_ids = [self.store._slot_to_id(s) for s in path]
            if all(pid is not None for pid in path_ids):
                result.append(path_ids)
        return Result(kind="paths", data=result, count=len(result))

    @handles(ShortestPathQuery)
    def _shortest_path(self, q: ShortestPathQuery) -> Result:
        src_slot = self._resolve_slot(q.from_id)
        tgt_slot = self._resolve_slot(q.to_id)
        if src_slot is None or tgt_slot is None:
            return Result(kind="path", data=None, count=0)
        edge_type = self._extract_kind_from_where(q.where)
        matrix = self.store.edge_matrices.get({edge_type} if edge_type else None)
        # Mask invisible nodes before BFS so the traversal routes around
        # hidden (retracted/expired/out-of-context) nodes instead of
        # returning None when the shortest candidate path happens to
        # include one (bug #39).
        matrix = self._visibility_masked_matrix(matrix)
        if matrix is None:
            return Result(kind="path", data=None, count=0)
        matrix_t = matrix.T.tocsr()
        # Pass q.max_depth through — None means unbounded. Previously this
        # call omitted the kwarg and picked up the old hardcoded default of
        # 10, silently capping shortest-path queries past 10 hops (bug #40).
        path = bidirectional_bfs(
            matrix, matrix_t, src_slot, tgt_slot, max_depth=q.max_depth,
        )
        if path is None:
            return Result(kind="path", data=None, count=0)
        path_ids = [self.store._slot_to_id(s) for s in path]
        if any(pid is None for pid in path_ids):
            return Result(kind="path", data=None, count=0)
        return Result(kind="path", data=path_ids, count=len(path_ids))

    @handles(DistanceQuery)
    def _distance(self, q: DistanceQuery) -> Result:
        src_slot = self._resolve_slot(q.from_id)
        tgt_slot = self._resolve_slot(q.to_id)
        if src_slot is None or tgt_slot is None:
            return Result(kind="distance", data=-1, count=1)
        matrix = self.store.edge_matrices.get(None)
        if matrix is None:
            if src_slot == tgt_slot:
                return Result(kind="distance", data=0, count=1)
            return Result(kind="distance", data=-1, count=1)
        matrix = self._visibility_masked_matrix(matrix)
        matrix_t = matrix.T.tocsr()
        path = bidirectional_bfs(matrix, matrix_t, src_slot, tgt_slot, q.max_depth)
        if path is not None:
            path_ids = [self.store._slot_to_id(s) for s in path]
            if any(pid is None for pid in path_ids):
                path = None
        dist = len(path) - 1 if path else -1
        return Result(kind="distance", data=dist, count=1)

    @handles(WeightedShortestPathQuery)
    def _weighted_shortest_path(self, q: WeightedShortestPathQuery) -> Result:
        src_slot = self._resolve_slot(q.from_id)
        tgt_slot = self._resolve_slot(q.to_id)
        if src_slot is None or tgt_slot is None:
            return Result(kind="path", data=None, count=0)
        edge_type = self._extract_kind_from_where(q.where)
        matrix = self.store.edge_matrices.get({edge_type} if edge_type else None)
        # Pre-mask so dijkstra routes around hidden nodes (bug #39 for
        # weighted variant).
        matrix = self._visibility_masked_matrix(matrix)
        if matrix is None:
            return Result(kind="path", data=None, count=0)
        max_cost = float(q.max_depth) if q.max_depth is not None else float("inf")
        path, cost = dijkstra(matrix, src_slot, tgt_slot, max_cost=max_cost)
        if path is None:
            return Result(kind="path", data=None, count=0)
        path_ids = [self.store._slot_to_id(s) for s in path]
        if any(pid is None for pid in path_ids):
            return Result(kind="path", data=None, count=0)
        return Result(kind="path", data=path_ids, count=len(path_ids))

    @handles(WeightedDistanceQuery)
    def _weighted_distance(self, q: WeightedDistanceQuery) -> Result:
        src_slot = self._resolve_slot(q.from_id)
        tgt_slot = self._resolve_slot(q.to_id)
        if src_slot is None or tgt_slot is None:
            return Result(kind="distance", data=-1, count=1)
        matrix = self.store.edge_matrices.get(None)
        if matrix is None:
            if src_slot == tgt_slot:
                return Result(kind="distance", data=0.0, count=1)
            return Result(kind="distance", data=-1, count=1)
        matrix = self._visibility_masked_matrix(matrix)
        max_cost = float(q.max_depth) if q.max_depth is not None else float("inf")
        path, cost = dijkstra(matrix, src_slot, tgt_slot, max_cost=max_cost)
        if path is not None:
            path_ids = [self.store._slot_to_id(s) for s in path]
            if any(pid is None for pid in path_ids):
                path = None
        if path is None:
            return Result(kind="distance", data=-1, count=1)
        return Result(kind="distance", data=cost, count=1)

    @handles(AncestorsQuery)
    def _ancestors(self, q: AncestorsQuery) -> Result:
        slot = self._resolve_slot(q.node_id)
        if slot is None:
            return Result(kind="subgraph", data={"nodes": [], "edges": []}, count=0)
        edge_type = self._extract_kind_from_where(q.where)
        if edge_type:
            matrix = self.store.edge_matrices.get_transpose(edge_type)
        else:
            matrix = self.store.edge_matrices.get(None)
            if matrix is not None:
                matrix = matrix.T.tocsr()
        if matrix is None:
            node_data = self.store.get_node(q.node_id)
            return Result(kind="subgraph", data={"nodes": [{**node_data, "_query_anchor": True}] if node_data else [], "edges": []}, count=0)
        visited = bfs_traverse(matrix, slot, q.depth)
        visited_slots = {node_idx for node_idx, _ in visited}
        nodes = []
        for node_idx, depth in visited:
            nid = self.store._slot_to_id(node_idx)
            if nid:
                node = self.store.get_node(nid)
                if node:
                    node["_depth"] = depth
                    if node_idx == slot:
                        node["_query_anchor"] = True
                    nodes.append(node)
        nodes = self._filter_visible(nodes)
        visible_ids = {n["id"] for n in nodes}
        edges = []
        for node_idx in visited_slots:
            nid = self.store._slot_to_id(node_idx)
            if nid and nid in visible_ids:
                for e in self.store.get_edges_from(nid, kind=edge_type):
                    if e["target"] in visible_ids:
                        edges.append(e)
        ancestor_count = sum(1 for n in nodes if not n.get("_query_anchor"))
        return Result(kind="subgraph", data={"nodes": nodes, "edges": edges}, count=ancestor_count)

    @handles(DescendantsQuery)
    def _descendants(self, q: DescendantsQuery) -> Result:
        slot = self._resolve_slot(q.node_id)
        if slot is None:
            return Result(kind="subgraph", data={"nodes": [], "edges": []}, count=0)
        edge_type = self._extract_kind_from_where(q.where)
        matrix = self.store.edge_matrices.get({edge_type} if edge_type else None)
        if matrix is None:
            node_data = self.store.get_node(q.node_id)
            return Result(kind="subgraph", data={"nodes": [{**node_data, "_query_anchor": True}] if node_data else [], "edges": []}, count=0)
        visited = bfs_traverse(matrix, slot, q.depth)
        visited_slots = {node_idx for node_idx, _ in visited}
        nodes = []
        for node_idx, depth in visited:
            nid = self.store._slot_to_id(node_idx)
            if nid:
                node = self.store.get_node(nid)
                if node:
                    node["_depth"] = depth
                    if node_idx == slot:
                        node["_query_anchor"] = True
                    nodes.append(node)
        nodes = self._filter_visible(nodes)
        visible_ids = {n["id"] for n in nodes}
        edges = []
        for node_idx in visited_slots:
            nid = self.store._slot_to_id(node_idx)
            if nid and nid in visible_ids:
                for e in self.store.get_edges_from(nid, kind=edge_type):
                    if e["target"] in visible_ids:
                        edges.append(e)
        descendant_count = sum(1 for n in nodes if not n.get("_query_anchor"))
        return Result(kind="subgraph", data={"nodes": nodes, "edges": edges}, count=descendant_count)

    @handles(CommonNeighborsQuery)
    def _common_neighbors(self, q: CommonNeighborsQuery) -> Result:
        slot_a = self._resolve_slot(q.node_a)
        slot_b = self._resolve_slot(q.node_b)
        if slot_a is None or slot_b is None:
            return Result(kind="nodes", data=[], count=0)
        edge_type = self._extract_kind_from_where(q.where)
        matrix = self.store.edge_matrices.get({edge_type} if edge_type else None)
        if matrix is None:
            return Result(kind="nodes", data=[], count=0)
        shared = common_neighbors(matrix, slot_a, slot_b)
        nodes = []
        for idx in shared:
            nid = self.store._slot_to_id(int(idx))
            if nid:
                node = self.store.get_node(nid)
                if node:
                    nodes.append(node)
        nodes = self._filter_visible(nodes)
        return Result(kind="nodes", data=nodes, count=len(nodes))
