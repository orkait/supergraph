
import logging
import sys

logger = logging.getLogger(__name__)
from supergraph.core.errors import CeilingExceeded
from supergraph.algos.measure import (
    estimate_bytes as _algo_estimate_bytes,
    will_exceed_ceiling as _algo_will_exceed,
    measure_components as _algo_measure_components,
    BYTES_PER_NODE_DEFAULT,
    BYTES_PER_EDGE_DEFAULT,
)

DEFAULT_CEILING_BYTES = 256 * 1_000_000

BYTES_PER_NODE_ESTIMATE = BYTES_PER_NODE_DEFAULT
BYTES_PER_EDGE_ESTIMATE = BYTES_PER_EDGE_DEFAULT

BYTES_PER_NODE = BYTES_PER_NODE_ESTIMATE
BYTES_PER_EDGE = BYTES_PER_EDGE_ESTIMATE


def estimate(node_count: int, edge_count: int) -> int:
    return _algo_estimate_bytes(node_count, edge_count)


def measure(store, vector_store=None, document_store=None, skip_csr: bool = False) -> dict:
    st = store.string_table
    node_arrays_bytes = store.node_ids.nbytes + store.node_kinds.nbytes

    str_list_size = sys.getsizeof(st._id_to_str)
    str_dict_size = sys.getsizeof(st._str_to_id)
    string_table_bytes = str_list_size + str_dict_size + len(st) * 20

    edge_lists_bytes = 0
    for etype, edges in store._edges_by_type.items():
        edge_lists_bytes += sys.getsizeof(edges) + len(edges) * 100

    edge_data_idx_bytes = 0
    for etype, idx in store._edge_data_idx.items():
        edge_data_idx_bytes += sys.getsizeof(idx) + len(idx) * 80

    edge_matrices_bytes = 0
    if not skip_csr and hasattr(store, '_edge_matrices'):
        store._ensure_edges_built()
        for m in store._edge_matrices._typed.values():
            edge_matrices_bytes += m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
        if store._edge_matrices._combined_all is not None:
            m = store._edge_matrices._combined_all
            edge_matrices_bytes += m.data.nbytes + m.indices.nbytes + m.indptr.nbytes

    edge_keys_bytes = sys.getsizeof(store._edge_keys) + len(store._edge_keys) * 80
    tombstones_bytes = sys.getsizeof(store.node_tombstones) + len(store.node_tombstones) * 28
    id_to_slot_bytes = sys.getsizeof(store.id_to_slot) + len(store.id_to_slot) * 40
    vector_bytes = vector_store.memory_bytes if vector_store is not None else 0

    report = _algo_measure_components(
        node_arrays_bytes=node_arrays_bytes,
        column_nbytes=store.columns.memory_bytes,
        string_table_bytes=string_table_bytes,
        edge_lists_bytes=edge_lists_bytes,
        edge_data_idx_bytes=edge_data_idx_bytes,
        edge_matrices_bytes=edge_matrices_bytes,
        edge_keys_bytes=edge_keys_bytes,
        tombstones_bytes=tombstones_bytes,
        id_to_slot_bytes=id_to_slot_bytes,
        vector_store_bytes=vector_bytes,
    )

    if document_store is not None:
        try:
            stats = document_store.stats()
            report["document_store_disk"] = stats.get("total_bytes", 0)
        except Exception as err:
            logger.debug("document_store.stats() failed during memory accounting: %s", err)

    return report


def check_ceiling(
    current_nodes: int,
    current_edges: int,
    added_nodes: int,
    added_edges: int,
    ceiling_bytes: int = DEFAULT_CEILING_BYTES,
    bytes_per_node: int | None = None,
    bytes_per_edge: int | None = None,
) -> None:
    bpn = bytes_per_node if bytes_per_node is not None else BYTES_PER_NODE_ESTIMATE
    bpe = bytes_per_edge if bytes_per_edge is not None else BYTES_PER_EDGE_ESTIMATE
    exceeds, current_mb = _algo_will_exceed(
        current_nodes=current_nodes,
        current_edges=current_edges,
        added_nodes=added_nodes,
        added_edges=added_edges,
        ceiling_bytes=ceiling_bytes,
        bytes_per_node=bpn,
        bytes_per_edge=bpe,
    )
    if exceeds:
        raise CeilingExceeded(
            current_mb=current_mb,
            ceiling_mb=ceiling_bytes // 1_000_000,
            operation=f"add {added_nodes} nodes, {added_edges} edges",
        )


def check_ceiling_accurate(store, vector_store, ceiling_bytes: int) -> bool:
    report = measure(store, vector_store)
    return report["total"] > ceiling_bytes
