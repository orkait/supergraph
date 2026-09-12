"""Edge-list primitives.

Pure operations on edge dicts ({etype: [(src, tgt, data), ...]}) and
scipy CSR matrices. The shell layer (core/store.py, core/optimizer.py)
applies the results to mutable EdgeMatrices / CoreStore state.
"""

import numpy as np
from scipy.sparse import csr_matrix

__all__ = [
    "resize_csr",
    "cascade_filter_edges",
    "rewire_edges_source_target",
    "dedupe_edges_by_src_tgt",
    "rebuild_edge_keys_set",
    "rebuild_edge_data_idx",
    "build_typed_csrs",
]


def resize_csr(mat: csr_matrix, n: int) -> csr_matrix:
    """Pad a CSR matrix's indptr to shape (n, n).

    Used when nodes are added after the last edge rebuild and the matrix
    trails _next_slot. Padding with the last indptr value marks the new
    rows as edge-free without a full rebuild.
    """
    if mat.shape[0] >= n:
        return mat
    old_size = len(mat.indptr)
    needed = n + 1
    if old_size < needed:
        pad = np.full(needed - old_size, mat.indptr[-1], dtype=mat.indptr.dtype)
        new_indptr = np.concatenate([mat.indptr, pad])
    else:
        new_indptr = mat.indptr
    return csr_matrix((mat.data, mat.indices, new_indptr), shape=(n, n))


def cascade_filter_edges(
    edges_by_type: dict,
    removed_slots: set,
) -> tuple:
    """Drop every edge whose src or tgt is in `removed_slots`.

    Args:
        edges_by_type: {etype: [(src, tgt, data), ...]}.
        removed_slots: set of slot indices to cascade-remove.

    Returns:
        (new_edges_by_type, any_removed) - new dict is filtered, empty
        etype lists are dropped. `any_removed` is True iff anything was
        filtered out.
    """
    if not removed_slots:
        return dict(edges_by_type), False
    new_edges: dict = {}
    any_removed = False
    for etype, edges in edges_by_type.items():
        kept = [(s, t, d) for s, t, d in edges if s not in removed_slots and t not in removed_slots]
        if len(kept) != len(edges):
            any_removed = True
        if kept:
            new_edges[etype] = kept
    return new_edges, any_removed


def rewire_edges_source_target(
    edges_by_type: dict,
    src_slot: int,
    tgt_slot: int,
) -> tuple:
    """Redirect every edge touching src_slot to tgt_slot (for MERGE).

    Args:
        edges_by_type: {etype: [(src, tgt, data), ...]}.
        src_slot: slot being merged away.
        tgt_slot: destination slot.

    Returns:
        (new_edges_by_type, rewire_count) - rewired dict and the number
        of edges whose src or tgt changed.
    """
    new_edges: dict = {}
    rewired = 0
    for etype, edges in edges_by_type.items():
        new_list = []
        for s, t, d in edges:
            # Self-loop on the source: ``src → src`` would rewire to
            # ``(tgt, src, d)`` under the old first-branch-wins logic,
            # leaving ``src`` as a target after MERGE deletes it — a
            # zombie edge pointing at a tombstoned slot (bug #100). Detect
            # and rewrite to ``(tgt, tgt, d)`` so the self-loop survives
            # MERGE pointing at the merged node.
            if s == src_slot and t == src_slot:
                new_list.append((tgt_slot, tgt_slot, d))
                rewired += 1
            elif s == src_slot:
                new_list.append((tgt_slot, t, d))
                rewired += 1
            elif t == src_slot:
                new_list.append((s, tgt_slot, d))
                rewired += 1
            else:
                new_list.append((s, t, d))
        new_edges[etype] = new_list
    return new_edges, rewired


def dedupe_edges_by_src_tgt(edges_by_type: dict) -> dict:
    """Drop duplicate (src, tgt) pairs within each edge type, first-wins.

    Returns a fresh dict; empty edge lists are removed.
    """
    result: dict = {}
    for etype, edges in edges_by_type.items():
        seen: set = set()
        deduped: list = []
        for s, t, d in edges:
            key = (s, t)
            if key not in seen:
                seen.add(key)
                deduped.append((s, t, d))
        if deduped:
            result[etype] = deduped
    return result


def rebuild_edge_keys_set(edges_by_type: dict) -> set:
    """Recompute the (src, tgt, etype) key set from scratch."""
    return {
        (s, t, k)
        for k, edges in edges_by_type.items()
        for s, t, _d in edges
    }


def rebuild_edge_data_idx(edges_by_type: dict) -> dict:
    """Recompute the {etype: {(src, tgt): data}} index from scratch."""
    return {
        k: {(s, t): d for s, t, d in edges}
        for k, edges in edges_by_type.items()
    }


def build_typed_csrs(
    edges_by_type: dict,
    num_nodes: int,
) -> tuple:
    """Build per-type CSR matrices + per-type edge data lists from raw edges.

    Single-pass src/tgt/weight extraction - one list scan per edge type
    instead of three list comprehensions.

    Args:
        edges_by_type: {etype: [(src, tgt, data_dict), ...]}.
        num_nodes: total slot count (shape dimension for every matrix).

    Returns:
        (typed_csrs, edge_data_lists) where
            typed_csrs: {etype: csr_matrix(shape=(num_nodes, num_nodes))}
            edge_data_lists: {etype: [data_dict, ...]} parallel to edges.
    """
    typed: dict = {}
    data_lists: dict = {}
    for etype, edge_list in edges_by_type.items():
        if not edge_list:
            continue
        m = len(edge_list)
        sources = np.empty(m, dtype=np.int32)
        targets = np.empty(m, dtype=np.int32)
        weights = np.empty(m, dtype=np.float32)
        data_list: list = [None] * m
        for i, (s, t, d) in enumerate(edge_list):
            sources[i] = s
            targets[i] = t
            weights[i] = d.get("weight", 1.0) if d else 1.0
            data_list[i] = d
        typed[etype] = csr_matrix(
            (weights, (sources, targets)),
            shape=(num_nodes, num_nodes),
        )
        data_lists[etype] = data_list
    return typed, data_lists
