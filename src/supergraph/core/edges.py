"""Typed edge matrices backed by scipy sparse CSR."""

import numpy as np
from scipy.sparse import csr_matrix

from supergraph.algos.edges_ops import (
    resize_csr,  # noqa: F401 - re-export
    build_typed_csrs,
)


from supergraph.core.compressed_edges import CompressedEdgeMatrix

class EdgeMatrices:
    def __init__(self, use_compression: bool = False):
        self._use_compression = use_compression
        self._typed: dict[str, csr_matrix | CompressedEdgeMatrix] = {}     # per-type CSR
        self._combined_all: csr_matrix | CompressedEdgeMatrix | None = None  # precomputed union
        self._cache: dict[frozenset, csr_matrix | CompressedEdgeMatrix] = {} # combination cache
        self._edge_data: dict[str, list[dict]] = {}  # per-type edge data lists
        self._transpose_cache: dict[str, csr_matrix | CompressedEdgeMatrix] = {} # CSC for incoming edges
        self._combined_transpose: csr_matrix | CompressedEdgeMatrix | None = None  # cached combined-all transpose
        self._combined_spread: csr_matrix | CompressedEdgeMatrix | None = None

        # Dynamic Edge Buffer (LSM L0)
        self._dynamic_out: dict[str, dict[int, list[int]]] = {}
        self._dynamic_in: dict[str, dict[int, list[int]]] = {}
        self._dynamic_weights: dict[str, dict[tuple[int, int], float]] = {}
        self._pending_edge_count = 0
        self._num_nodes = 0
        
        # Caches for dynamic combination (Frozen + Delta)
        self._dynamic_cache: dict[frozenset | str | None, csr_matrix] = {}
        self._dynamic_transpose_cache: dict[str, csr_matrix] = {}
        self._dynamic_combined_transpose: csr_matrix | None = None
        self._dynamic_combined_spread: csr_matrix | None = None

        # Degree arrays - precomputed on rebuild
        self._out_degree: dict[str, np.ndarray] = {}
        self._out_degree_all: np.ndarray | None = None
        self._in_degree: dict[str, np.ndarray] = {}

    @property
    def edge_types(self) -> list[str]:
        """Return list of edge type names."""
        types = set(self._typed.keys())
        types.update(self._dynamic_out.keys())
        return list(types)

    @property
    def total_edges(self) -> int:
        """Total number of edges across all types."""
        return sum(m.nnz for m in self._typed.values()) + self._pending_edge_count

    def add_dynamic(self, src: int, tgt: int, kind: str, num_nodes: int, weight: float = 1.0):
        self._dynamic_out.setdefault(kind, {}).setdefault(src, []).append(tgt)
        self._dynamic_in.setdefault(kind, {}).setdefault(tgt, []).append(src)
        self._dynamic_weights.setdefault(kind, {})[(src, tgt)] = weight
        self._pending_edge_count += 1
        self._num_nodes = max(self._num_nodes, num_nodes)
        
        # Clear only dynamic caches, preserve frozen matrices and transposes
        self._dynamic_cache.clear()
        self._dynamic_combined_transpose = None
        self._dynamic_combined_spread = None
        self._dynamic_transpose_cache.clear()

    def get(self, edge_types: set[str] | None = None) -> csr_matrix | CompressedEdgeMatrix | None:
        """Get matrix for query. None = all types. Merges Delta CSR if dynamic edges exist."""
        # Check dynamic cache first
        dyn_key = None if edge_types is None else (next(iter(edge_types)) if len(edge_types) == 1 else frozenset(edge_types))
        if dyn_key in self._dynamic_cache:
            return self._dynamic_cache[dyn_key]

        base = None
        if edge_types is None:
            base = self._combined_all
        elif len(edge_types) == 1:
            etype = next(iter(edge_types))
            base = self._typed.get(etype)
        else:
            key = frozenset(edge_types)
            if key not in self._cache:
                matrices = [self._typed[t] for t in edge_types if t in self._typed]
                if matrices:
                    if self._use_compression:
                        res = matrices[0]
                        for m in matrices[1:]:
                            res = res + m
                        self._cache[key] = res
                    else:
                        self._cache[key] = sum(matrices)
            base = self._cache.get(key)

        if self._pending_edge_count == 0:
            self._dynamic_cache[dyn_key] = base
            return base

        # Build Delta CSR
        srcs = []
        tgts = []
        data_list = []
        types_to_check = self._dynamic_out.keys() if edge_types is None else edge_types
        for etype in types_to_check:
            if etype in self._dynamic_out:
                for src, tgts_list in self._dynamic_out[etype].items():
                    for tgt in tgts_list:
                        srcs.append(src)
                        tgts.append(tgt)
                        data_list.append(self._dynamic_weights.get(etype, {}).get((src, tgt), 1.0))

        if not srcs:
            self._dynamic_cache[dyn_key] = base
            return base

        data = np.array(data_list, dtype=np.float32)
        n = self._num_nodes
        if base is not None:
            n = max(n, base.shape[0])
            
        delta = csr_matrix((data, (srcs, tgts)), shape=(n, n))
        
        if base is None:
            result = delta
        else:
            # Pad base if necessary
            if not isinstance(base, CompressedEdgeMatrix) and base.shape[0] < n:
                base = resize_csr(base, n)
            result = base + delta
            
        self._dynamic_cache[dyn_key] = result
        return result

    def get_transpose(self, edge_type: str) -> csr_matrix | CompressedEdgeMatrix | None:
        """Get transposed matrix for incoming edge queries."""
        if edge_type in self._dynamic_transpose_cache:
            return self._dynamic_transpose_cache[edge_type]

        base = None
        if edge_type in self._typed:
            if edge_type not in self._transpose_cache:
                self._transpose_cache[edge_type] = self._typed[edge_type].T
                if not isinstance(self._transpose_cache[edge_type], (csr_matrix, CompressedEdgeMatrix)):
                    # Handle case where .T returns a different type (unlikely but safe)
                    self._transpose_cache[edge_type] = self._transpose_cache[edge_type].tocsr()
            base = self._transpose_cache[edge_type]

        if self._pending_edge_count == 0:
            self._dynamic_transpose_cache[edge_type] = base
            return base

        srcs = []
        tgts = []
        data_list = []
        if edge_type in self._dynamic_in:
            for tgt, srcs_list in self._dynamic_in[edge_type].items():
                for src in srcs_list:
                    tgts.append(tgt)
                    srcs.append(src)
                    data_list.append(self._dynamic_weights.get(edge_type, {}).get((src, tgt), 1.0))

        if not srcs:
            self._dynamic_transpose_cache[edge_type] = base
            return base

        data = np.array(data_list, dtype=np.float32)
        n = self._num_nodes
        if base is not None:
            n = max(n, base.shape[0])
            
        delta = csr_matrix((data, (tgts, srcs)), shape=(n, n))

        if base is None:
            result = delta
        else:
            if not isinstance(base, CompressedEdgeMatrix) and base.shape[0] < n:
                base = resize_csr(base, n)
            result = base + delta

        self._dynamic_transpose_cache[edge_type] = result
        return result

    def get_combined_transpose_split(self) -> tuple[csr_matrix | CompressedEdgeMatrix | None, csr_matrix | None]:
        """Cached transpose of combined-all matrix. Returns (base, delta) to avoid O(N) merge."""
        base = None
        if self._combined_all is not None:
            if self._combined_transpose is None:
                self._combined_transpose = self._combined_all.T
            base = self._combined_transpose

        if self._pending_edge_count == 0:
            return base, None

        srcs = []
        tgts = []
        data_list = []
        for etype in self._dynamic_in:
            for tgt, srcs_list in self._dynamic_in[etype].items():
                for src in srcs_list:
                    tgts.append(tgt)
                    srcs.append(src)
                    data_list.append(self._dynamic_weights.get(etype, {}).get((src, tgt), 1.0))

        if not srcs:
            return base, None

        data = np.array(data_list, dtype=np.float32)
        n = self._num_nodes
        if base is not None:
            n = max(n, base.shape[0])

        delta = csr_matrix((data, (tgts, srcs)), shape=(n, n))

        if base is not None and not isinstance(base, CompressedEdgeMatrix) and base.shape[0] < n:
            base = resize_csr(base, n)

        return base, delta

    def get_combined_transpose(self) -> csr_matrix | CompressedEdgeMatrix | None:
        """Cached transpose of combined-all matrix."""
        if self._dynamic_combined_transpose is not None:
            return self._dynamic_combined_transpose

        base = None
        if self._combined_all is not None:
            if self._combined_transpose is None:
                self._combined_transpose = self._combined_all.T
            base = self._combined_transpose

        if self._pending_edge_count == 0:
            self._dynamic_combined_transpose = base
            return base

        srcs = []
        tgts = []
        data_list = []
        for etype in self._dynamic_in:
            for tgt, srcs_list in self._dynamic_in[etype].items():
                for src in srcs_list:
                    tgts.append(tgt)
                    srcs.append(src)
                    data_list.append(self._dynamic_weights.get(etype, {}).get((src, tgt), 1.0))

        if not srcs:
            self._dynamic_combined_transpose = base
            return base

        data = np.array(data_list, dtype=np.float32)
        n = self._num_nodes
        if base is not None:
            n = max(n, base.shape[0])

        delta = csr_matrix((data, (tgts, srcs)), shape=(n, n))
        
        if base is None:
            result = delta
        else:
            if not isinstance(base, CompressedEdgeMatrix) and base.shape[0] < n:
                base = resize_csr(base, n)
            result = base + delta
            
        self._dynamic_combined_transpose = result
        return result

    def get_combined_spread_split(self) -> tuple[csr_matrix | CompressedEdgeMatrix | None, csr_matrix | None]:
        """Return propagation matrices for bidirectional spreading."""
        base = None
        if self._combined_all is not None:
            if self._combined_spread is None:
                if self._combined_transpose is None:
                    self._combined_transpose = self._combined_all.T
                self._combined_spread = self._combined_all + self._combined_transpose
            base = self._combined_spread

        if self._pending_edge_count == 0:
            return base, None

        srcs = []
        tgts = []
        data_list = []
        for etype in self._dynamic_out:
            weights = self._dynamic_weights.get(etype, {})
            for src, tgts_list in self._dynamic_out[etype].items():
                for tgt in tgts_list:
                    weight = weights.get((src, tgt), 1.0)
                    srcs.extend((src, tgt))
                    tgts.extend((tgt, src))
                    data_list.extend((weight, weight))

        if not srcs:
            return base, None

        data = np.array(data_list, dtype=np.float32)
        n = self._num_nodes
        if base is not None:
            n = max(n, base.shape[0])

        delta = csr_matrix((data, (srcs, tgts)), shape=(n, n))

        if base is not None and not isinstance(base, CompressedEdgeMatrix) and base.shape[0] < n:
            base = resize_csr(base, n)

        return base, delta


    def get_combined_spread(self) -> csr_matrix | None:
        """Cached bidirectional propagation matrix for spreading activation."""
        if self._dynamic_combined_spread is not None:
            return self._dynamic_combined_spread

        base, delta = self.get_combined_spread_split()
        if delta is None:
            self._dynamic_combined_spread = base
            return base

        if base is None:
            result = delta
        else:
            if base.shape[0] < delta.shape[0]:
                base = resize_csr(base, delta.shape[0])
            result = base + delta

        self._dynamic_combined_spread = result
        return result

    def get_edge_data(self, edge_type: str) -> list[dict]:
        """Get edge data for a given type."""
        return self._edge_data.get(edge_type, [])

    def out_degree(self, edge_type: str | None = None) -> np.ndarray | None:
        """Get out-degree array. None = all types combined."""
        matrix = self.get({edge_type} if edge_type else None)
        if matrix is None:
            return None
        return np.diff(matrix.indptr)

    def in_degree(self, edge_type: str) -> np.ndarray | None:
        """Get in-degree array for given type."""
        t = self.get_transpose(edge_type)
        if t is None:
            return None
        return np.diff(t.indptr)

    def neighbors_out(self, node_idx: int, edge_type: str | None = None) -> np.ndarray:
        """Get outgoing neighbor indices for a node."""
        base = np.array([], dtype=np.int32)
        
        base_matrix = None
        if edge_type is None:
            base_matrix = self._combined_all
        elif len({edge_type}) == 1:
            base_matrix = self._typed.get(edge_type)

        if base_matrix is not None and node_idx < base_matrix.shape[0]:
            if isinstance(base_matrix, CompressedEdgeMatrix):
                base = base_matrix.get_row(node_idx)
            else:
                start = base_matrix.indptr[node_idx]
                end = base_matrix.indptr[node_idx + 1]
                base = base_matrix.indices[start:end]

        dynamic = []
        if edge_type is None:
            for etype, dyn in self._dynamic_out.items():
                dynamic.extend(dyn.get(node_idx, []))
        else:
            dynamic.extend(self._dynamic_out.get(edge_type, {}).get(node_idx, []))

        if not dynamic:
            return base.copy() if len(base) > 0 else base

        return np.concatenate((base, np.array(dynamic, dtype=np.int32)))

    def neighbors_in(self, node_idx: int, edge_type: str) -> np.ndarray:
        """Get incoming neighbor indices for a node."""
        base = np.array([], dtype=np.int32)
        
        base_matrix = None
        if edge_type in self._typed:
            if edge_type not in self._transpose_cache:
                self._transpose_cache[edge_type] = self._typed[edge_type].T.tocsr()
            base_matrix = self._transpose_cache[edge_type]

        if base_matrix is not None and node_idx < base_matrix.shape[0]:
            if isinstance(base_matrix, CompressedEdgeMatrix):
                base = base_matrix.get_row(node_idx)
            else:
                start = base_matrix.indptr[node_idx]
                end = base_matrix.indptr[node_idx + 1]
                base = base_matrix.indices[start:end]

        dynamic = []
        dynamic.extend(self._dynamic_in.get(edge_type, {}).get(node_idx, []))

        if not dynamic:
            return base.copy() if len(base) > 0 else base

        return np.concatenate((base, np.array(dynamic, dtype=np.int32)))

    def rebuild(self, edges_by_type: dict[str, list[tuple]], num_nodes: int):
        """Full rebuild from edge lists. Delegates CSR construction to algos."""
        self._typed.clear()
        self._cache.clear()
        self._transpose_cache.clear()
        self._combined_transpose = None
        self._combined_spread = None
        self._edge_data.clear()
        self._out_degree.clear()
        self._in_degree.clear()
        
        self._dynamic_out.clear()
        self._dynamic_in.clear()
        self._dynamic_weights.clear()
        self._dynamic_cache.clear()
        self._dynamic_transpose_cache.clear()
        self._dynamic_combined_transpose = None
        self._dynamic_combined_spread = None
        self._pending_edge_count = 0
        self._num_nodes = num_nodes

        typed, data_lists = build_typed_csrs(edges_by_type, num_nodes)
        
        if self._use_compression:
            for etype in list(typed.keys()):
                typed[etype] = CompressedEdgeMatrix.from_csr(typed[etype])
        
        self._typed = typed
        self._edge_data = data_lists

        if self._typed:
            if self._use_compression:
                # Merge all typed matrices into combined_all
                matrices = list(self._typed.values())
                res = matrices[0]
                for m in matrices[1:]:
                    res = res + m
                self._combined_all = res
            else:
                self._combined_all = sum(self._typed.values())
        else:
            self._combined_all = None

        for etype, m in self._typed.items():
            if isinstance(m, CompressedEdgeMatrix):
                self._out_degree[etype] = m._out_degree
            else:
                self._out_degree[etype] = np.diff(m.indptr)
                
        if self._combined_all is not None:
            if isinstance(self._combined_all, CompressedEdgeMatrix):
                self._out_degree_all = self._combined_all._out_degree
            else:
                self._out_degree_all = np.diff(self._combined_all.indptr)
        else:
            self._out_degree_all = None

        for etype in list(self._typed.keys()):
            t = self.get_transpose(etype)
            if t is not None:
                if isinstance(t, CompressedEdgeMatrix):
                    self._in_degree[etype] = t._out_degree
                else:
                    self._in_degree[etype] = np.diff(t.indptr)

