"""Memory-efficient edge matrix using Delta-VByte compressed adjacency lists."""

import numpy as np
from supergraph.algos.compressed import pack_delta_vbyte, unpack_delta_vbyte

class CompressedEdgeMatrix:
    """Compressed adjacency matrix replacement for scipy.sparse.csr_matrix.
    
    Stores edges in variable-length Delta-VByte buffers.
    """

    def __init__(self, num_nodes: int):
        self.shape = (num_nodes, num_nodes)
        self._num_nodes = num_nodes
        self._row_offsets = np.zeros(num_nodes + 1, dtype=np.uint32)
        self._buffer = bytearray()
        self._out_degree = np.zeros(num_nodes, dtype=np.int32)
        self._total_edges = 0

    @classmethod
    def from_csr(cls, mat) -> "CompressedEdgeMatrix":
        """Convert a scipy CSR matrix to a CompressedEdgeMatrix."""
        inst = cls(mat.shape[0])
        ptr = mat.indptr
        idx = mat.indices
        
        current_offset = 0
        for i in range(inst._num_nodes):
            row_start = ptr[i]
            row_end = ptr[i+1]
            row_indices = idx[row_start:row_end]
            
            inst._row_offsets[i] = current_offset
            inst._out_degree[i] = len(row_indices)
            
            if len(row_indices) > 0:
                # Delta-VByte pack the row
                packed = pack_delta_vbyte(row_indices)
                inst._buffer.extend(packed)
                current_offset += len(packed)
                inst._total_edges += len(row_indices)
                
        inst._row_offsets[inst._num_nodes] = current_offset
        return inst

    @property
    def indptr(self) -> np.ndarray:
        return np.concatenate(([0], np.cumsum(self._out_degree)))

    def get_row(self, row_idx: int) -> np.ndarray:
        """Get all target indices for a specific row."""
        if row_idx >= self._num_nodes:
            return np.array([], dtype=np.int32)
            
        count = self._out_degree[row_idx]
        if count == 0:
            return np.array([], dtype=np.int32)
            
        offset = self._row_offsets[row_idx]
        end_offset = self._row_offsets[row_idx+1]
        
        data = self._buffer[offset:end_offset]
        return unpack_delta_vbyte(data, count).astype(np.int32)

    def dot(self, vector: np.ndarray) -> np.ndarray:
        """Sparse Matrix-Vector Multiplication (SpMV)."""
        result = np.zeros(self._num_nodes, dtype=np.float32)
        active_sources = np.nonzero(vector)[0]
        
        for i in active_sources:
            val = vector[i]
            targets = self.get_row(i)
            if len(targets) > 0:
                result[targets] += val
                
        return result

    @property
    def T(self) -> "CompressedEdgeMatrix":
        from scipy.sparse import csr_matrix
        indices = []
        for i in range(self._num_nodes):
            row = self.get_row(i)
            if len(row) > 0:
                indices.append(row)
        
        if not indices:
            return CompressedEdgeMatrix(self._num_nodes)
            
        all_indices = np.concatenate(indices)
        all_data = np.ones(len(all_indices), dtype=np.float32)
        csr = csr_matrix((all_data, all_indices, self.indptr), shape=self.shape)
        return CompressedEdgeMatrix.from_csr(csr.T.tocsr())

    def __add__(self, other) -> "CompressedEdgeMatrix":
        from scipy.sparse import csr_matrix
        
        def to_csr(m):
            if isinstance(m, csr_matrix):
                return m
            idxs = []
            for i in range(m._num_nodes):
                idxs.append(m.get_row(i))
            if not idxs:
                return csr_matrix(m.shape)
            all_idxs = np.concatenate(idxs)
            all_data = np.ones(len(all_idxs), dtype=np.float32)
            return csr_matrix((all_data, all_idxs, m.indptr), shape=m.shape)

        c1 = to_csr(self)
        c2 = to_csr(other)
        return CompressedEdgeMatrix.from_csr(c1 + c2)

    def tocsr(self) -> "csr_matrix":
        from scipy.sparse import csr_matrix
        idxs = []
        for i in range(self._num_nodes):
            idxs.append(self.get_row(i))
        if not idxs:
            return csr_matrix(self.shape)
        all_idxs = np.concatenate(idxs)
        all_data = np.ones(len(all_idxs), dtype=np.float32)
        return csr_matrix((all_data, all_idxs, self.indptr), shape=self.shape)

    @property
    def nnz(self) -> int:
        return self._total_edges

    @property
    def nbytes(self) -> int:
        """Total memory footprint of the compressed matrix."""
        return self._row_offsets.nbytes + len(self._buffer)
