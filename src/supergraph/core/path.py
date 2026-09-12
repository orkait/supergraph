"""Deprecation shim. Import from supergraph.algos.graph instead."""

from supergraph.algos.graph import (
    bfs_traverse,
    bidirectional_bfs,
    common_neighbors,
    dijkstra,
    find_all_paths,
)

__all__ = [
    "bfs_traverse",
    "bidirectional_bfs",
    "common_neighbors",
    "dijkstra",
    "find_all_paths",
]
