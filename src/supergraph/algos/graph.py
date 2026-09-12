"""Graph algorithms on scipy CSR matrices.

All functions operate on immutable matrix inputs. No supergraph imports.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as _csgraph_dijkstra

__all__ = [
    "bfs_traverse",
    "bfs_reach",
    "bidirectional_bfs",
    "find_all_paths",
    "dijkstra",
    "common_neighbors",
    "propagate_values",
]


def _transpose_csr_cached(matrix: csr_matrix) -> csr_matrix:
    cached = getattr(matrix, "_supergraph_transpose_csr", None)
    if cached is None or cached.shape != matrix.shape:
        cached = matrix.T.tocsr()
        matrix._supergraph_transpose_csr = cached
    return cached


def bidirectional_bfs(
    matrix: csr_matrix,
    matrix_t: csr_matrix,
    source: int,
    target: int,
    max_depth: int | None = None,
) -> list[int] | None:
    """Shortest unweighted path via bidirectional BFS.

    Returns list of node indices including source and target, or None.

    Implementation: dense boolean visited + int32 parent arrays per side.
    Each expansion gathers all neighbors of the current frontier in one
    numpy slice against CSR indptr/indices, filters unseen via boolean
    mask, and scatters parents with numpy.unique first-occurrence. Distance
    is tracked in an int32 array - no Python dict churn.
    """
    if source == target:
        return [source]

    n = max(matrix.shape[0], matrix_t.shape[0])
    if source >= n or target >= n or source < 0 or target < 0:
        return None

    # -1 sentinel = unvisited. Source/target are roots - parent=-2 marks them
    # as visited without a real parent (distinguishes from unvisited -1).
    _ROOT = -2
    fwd_parent = np.full(n, -1, dtype=np.int64)
    bwd_parent = np.full(n, -1, dtype=np.int64)
    fwd_dist = np.full(n, -1, dtype=np.int32)
    bwd_dist = np.full(n, -1, dtype=np.int32)
    fwd_parent[source] = _ROOT
    bwd_parent[target] = _ROOT
    fwd_dist[source] = 0
    bwd_dist[target] = 0

    fwd_frontier = np.array([source], dtype=np.int64)
    bwd_frontier = np.array([target], dtype=np.int64)

    step = 0
    while True:
        if max_depth is not None and step >= max_depth:
            return None
        step += 1

        if fwd_frontier.size == 0 and bwd_frontier.size == 0:
            return None

        expand_fwd = (
            fwd_frontier.size > 0
            and (bwd_frontier.size == 0 or fwd_frontier.size <= bwd_frontier.size)
        )
        if expand_fwd:
            fwd_frontier = _expand_frontier_np(
                matrix, fwd_frontier, fwd_parent, fwd_dist, step,
            )
        else:
            bwd_frontier = _expand_frontier_np(
                matrix_t, bwd_frontier, bwd_parent, bwd_dist, step,
            )

        # Meeting points: slots visited by both sides.
        both = (fwd_parent >= -1) & (fwd_parent != -1) & (bwd_parent != -1)
        # The first guard (>=-1) is a tautology on int64 but kept explicit;
        # what we need is "not unvisited", i.e. != -1.
        both = (fwd_parent != -1) & (bwd_parent != -1)
        if both.any():
            meeting_slots = np.nonzero(both)[0]
            combined = fwd_dist[meeting_slots].astype(np.int64) + bwd_dist[meeting_slots].astype(np.int64)
            # Tie-break: min combined distance, then min slot index for determinism.
            order = np.lexsort((meeting_slots, combined))
            mid = int(meeting_slots[order[0]])
            return _reconstruct_path_np(fwd_parent, bwd_parent, mid, source, target, _ROOT)


def _expand_frontier_np(
    matrix: csr_matrix,
    frontier: np.ndarray,
    parent: np.ndarray,
    dist: np.ndarray,
    cur_step: int,
) -> np.ndarray:
    """Vectorised one-hop expansion.

    Collects all neighbors of ``frontier`` via CSR row slicing, filters
    those already visited, picks the first parent per new neighbor, and
    scatters parents + distances in one numpy pass. Returns the next
    frontier as an int64 array.
    """
    if frontier.size == 0:
        return frontier

    indptr = matrix.indptr
    indices = matrix.indices

    # Per-row neighbor counts -> parent-per-neighbor via np.repeat.
    starts = indptr[frontier]
    ends = indptr[frontier + 1]
    counts = ends - starts
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)

    # Flat neighbor array: concatenate each row's slice.
    out = np.empty(total, dtype=indices.dtype)
    pos = 0
    # Small Python loop over |frontier|, not |nnz|: each iteration is a
    # vectorised slice assign. O(frontier) Python calls, O(nnz_of_frontier)
    # memory bandwidth - negligible compared to the old per-edge loop.
    for i, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
        c = e - s
        if c:
            out[pos:pos + c] = indices[s:e]
            pos += c
    parents_flat = np.repeat(frontier, counts)

    # Filter already-visited. np.unique with return_index keeps the first
    # parent per new neighbor so BFS parent pointers match a standard
    # level-ordered expansion.
    unseen = parent[out] == -1
    if not unseen.any():
        return np.empty(0, dtype=np.int64)

    new_nb = out[unseen]
    new_par = parents_flat[unseen]
    uniq, first_idx = np.unique(new_nb, return_index=True)
    parent[uniq] = new_par[first_idx]
    dist[uniq] = cur_step
    return uniq.astype(np.int64)


def _reconstruct_path_np(
    fwd_parent: np.ndarray,
    bwd_parent: np.ndarray,
    meeting_point: int,
    source: int,
    target: int,
    root_sentinel: int,
) -> list[int]:
    """Walk the parent arrays back to source and forward to target."""
    fwd_path: list[int] = []
    node = meeting_point
    while node != root_sentinel and node >= 0:
        fwd_path.append(int(node))
        p = int(fwd_parent[node])
        if p == root_sentinel or p < 0:
            break
        node = p
    fwd_path.reverse()

    bwd_path: list[int] = []
    node = int(bwd_parent[meeting_point])
    while node != root_sentinel and node >= 0:
        bwd_path.append(int(node))
        p = int(bwd_parent[node])
        if p == root_sentinel or p < 0:
            break
        node = p
    return fwd_path + bwd_path


def bfs_traverse(
    matrix: csr_matrix, start: int, max_depth: int
) -> list[tuple[int, int]]:
    """BFS from start up to max_depth via sparse matrix-power.

    One CSR.T @ bool matvec per hop - O(nnz) per level, vectorized.
    Returns (node_index, depth) pairs including (start, 0).
    """
    n = matrix.shape[0]
    result = [(start, 0)]
    if n == 0 or start >= n or max_depth <= 0:
        return result

    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    frontier = np.zeros(n, dtype=np.float32)
    frontier[start] = 1.0
    matT = _transpose_csr_cached(matrix)

    for depth in range(1, max_depth + 1):
        reached = matT.dot(frontier) > 0
        next_mask = reached & ~visited
        if not next_mask.any():
            break
        visited |= next_mask
        new_slots = np.nonzero(next_mask)[0]
        d_tuple = int(depth)
        result.extend((int(s), d_tuple) for s in new_slots)
        frontier = next_mask.astype(np.float32)

    return result


def find_all_paths(
    matrix: csr_matrix,
    source: int,
    target: int,
    max_depth: int,
    max_results: int = 100,
) -> list[list[int]]:
    """Enumerate all simple paths from source to target up to max_depth.

    Caps at max_results paths. Pure recursive DFS.
    """
    paths: list[list[int]] = []

    def dfs(node, path, visited_set, depth):
        if len(paths) >= max_results:
            return
        if node == target:
            paths.append(list(path))
            return
        if depth >= max_depth:
            return

        start_idx = matrix.indptr[node]
        end_idx = matrix.indptr[node + 1]
        neighbors = matrix.indices[start_idx:end_idx]

        for nb in neighbors:
            nb = int(nb)
            if nb not in visited_set:
                path.append(nb)
                visited_set.add(nb)
                dfs(nb, path, visited_set, depth + 1)
                path.pop()
                visited_set.discard(nb)

    dfs(source, [source], {source}, 0)
    return paths


def dijkstra(
    matrix: csr_matrix,
    source: int,
    target: int,
    max_cost: float = float('inf'),
) -> tuple[list[int] | None, float]:
    """Weighted shortest path via scipy.sparse.csgraph.dijkstra (native C)."""
    if source == target:
        return [source], 0.0

    limit = max_cost if max_cost != float('inf') else np.inf
    dist_arr, predecessors = _csgraph_dijkstra(
        matrix,
        indices=source,
        return_predecessors=True,
        limit=limit,
        directed=True,
    )
    total = float(dist_arr[target])
    if not np.isfinite(total):
        return None, float('inf')

    path: list[int] = []
    node = int(target)
    while node != source:
        path.append(node)
        node = int(predecessors[node])
        if node == -9999:
            return None, float('inf')
    path.append(source)
    path.reverse()
    return path, total


def common_neighbors(matrix: csr_matrix, node_a: int, node_b: int) -> np.ndarray:
    """Outgoing neighbors shared between two nodes."""
    start_a, end_a = matrix.indptr[node_a], matrix.indptr[node_a + 1]
    start_b, end_b = matrix.indptr[node_b], matrix.indptr[node_b + 1]
    neighbors_a = matrix.indices[start_a:end_a]
    neighbors_b = matrix.indices[start_b:end_b]
    return np.intersect1d(neighbors_a, neighbors_b)


def bfs_reach(
    matrix: csr_matrix,
    source: int,
    max_depth: int | None = None,
) -> set:
    """Set of slots reachable from source within max_depth hops (or unlimited)."""
    n = matrix.shape[0]
    if n == 0 or source >= n:
        return {source} if source >= 0 else set()

    visited: set = {source}
    frontier = np.zeros(n, dtype=np.float32)
    if source < n:
        frontier[source] = 1.0
    else:
        return visited

    matT = _transpose_csr_cached(matrix)
    depth = 0
    while True:
        if max_depth is not None and depth >= max_depth:
            break
        reached = matT.dot(frontier) > 0
        next_idx = np.nonzero(reached)[0]
        new_slots = [int(s) for s in next_idx if int(s) not in visited]
        if not new_slots:
            break
        visited.update(new_slots)
        next_mask = np.zeros(n, dtype=np.float32)
        next_mask[new_slots] = 1.0
        frontier = next_mask
        depth += 1
    return visited


def propagate_values(
    matrix: csr_matrix,
    source: int,
    source_value: float,
    depth: int,
    existing_values: np.ndarray,
    presence: np.ndarray,
    blocked_slots=None,
) -> tuple:
    """BFS forward, multiplying source_value by edge weights at each hop.

    Args:
        matrix: CSR weight matrix. matrix[i, j] == weight of edge i→j.
        source: starting slot.
        source_value: initial value at source.
        depth: max hops (inclusive).
        existing_values: per-slot current value array (read-only here).
        presence: per-slot bool mask of which slots have existing values.
        blocked_slots: optional set of slot indices to skip entirely.

    Returns:
        (updated_slots, new_values) - lists giving the slot indices and
        newly computed values in order. Caller writes back to columns.

    Note:
        This implements the semantics of `_propagate`: value diffuses
        multiplicatively along edge weights, each neighbor is visited
        at most once (BFS visited set), blocked slots are not traversed.
    """
    from collections import deque

    blocked = blocked_slots or set()
    visited: set = {source}
    frontier: deque = deque([(source, float(source_value), 0)])
    updated_slots: list = []
    new_values: list = []

    while frontier:
        current_slot, parent_value, current_depth = frontier.popleft()
        if current_depth >= depth:
            continue

        start = matrix.indptr[current_slot]
        end = matrix.indptr[current_slot + 1]
        neighbors = matrix.indices[start:end]
        weights = matrix.data[start:end]

        for i, nb in enumerate(neighbors):
            nb = int(nb)
            if nb in visited:
                continue
            visited.add(nb)
            if nb in blocked:
                continue
            edge_weight = float(weights[i]) if i < len(weights) else 1.0
            propagated = parent_value * edge_weight
            updated_slots.append(nb)
            new_values.append(propagated)
            frontier.append((nb, propagated, current_depth + 1))

    return updated_slots, new_values
