
__all__ = [
    "BYTES_PER_NODE_DEFAULT",
    "BYTES_PER_EDGE_DEFAULT",
    "estimate_bytes",
    "will_exceed_ceiling",
    "measure_components",
]

BYTES_PER_NODE_DEFAULT = 330
BYTES_PER_EDGE_DEFAULT = 20


def estimate_bytes(
    node_count: int,
    edge_count: int,
    bytes_per_node: int = BYTES_PER_NODE_DEFAULT,
    bytes_per_edge: int = BYTES_PER_EDGE_DEFAULT,
) -> int:
    return node_count * bytes_per_node + edge_count * bytes_per_edge


def will_exceed_ceiling(
    current_nodes: int,
    current_edges: int,
    added_nodes: int,
    added_edges: int,
    ceiling_bytes: int,
    bytes_per_node: int = BYTES_PER_NODE_DEFAULT,
    bytes_per_edge: int = BYTES_PER_EDGE_DEFAULT,
) -> tuple[bool, int]:
    projected = estimate_bytes(
        current_nodes + added_nodes,
        current_edges + added_edges,
        bytes_per_node,
        bytes_per_edge,
    )
    current_mb = estimate_bytes(
        current_nodes, current_edges, bytes_per_node, bytes_per_edge
    ) // 1_000_000
    return projected > ceiling_bytes, current_mb


def measure_components(
    node_arrays_bytes: int,
    column_nbytes: int,
    string_table_bytes: int,
    edge_lists_bytes: int,
    edge_data_idx_bytes: int,
    edge_matrices_bytes: int,
    edge_keys_bytes: int,
    tombstones_bytes: int,
    id_to_slot_bytes: int,
    vector_store_bytes: int = 0,
) -> dict:
    report: dict = {
        "node_arrays": int(node_arrays_bytes),
        "columns": int(column_nbytes),
        "string_table": int(string_table_bytes),
        "edge_lists": int(edge_lists_bytes),
        "edge_data_idx": int(edge_data_idx_bytes),
        "edge_matrices": int(edge_matrices_bytes),
        "edge_keys": int(edge_keys_bytes),
        "tombstones": int(tombstones_bytes),
        "id_to_slot": int(id_to_slot_bytes),
        "vector_store": int(vector_store_bytes),
    }
    core_total = sum(report.values())
    report["core_total"] = int(core_total)
    report["document_store_disk"] = 0
    report["total"] = int(core_total)
    return report
