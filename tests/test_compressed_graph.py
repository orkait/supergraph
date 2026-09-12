import pytest
import numpy as np
from supergraph import SuperGraph
from supergraph.core.compressed_edges import CompressedEdgeMatrix

def test_compressed_edge_storage():
    # 1. Initialize with compression enabled
    gs = SuperGraph(use_compression=True)
    
    # 2. Add some nodes and edges
    gs.execute('CREATE NODE "a" kind="person"')
    gs.execute('CREATE NODE "b" kind="person"')
    gs.execute('CREATE NODE "c" kind="person"')
    
    gs.execute('CREATE EDGE "a" -> "b" kind="knows"')
    gs.execute('CREATE EDGE "b" -> "c" kind="knows"')
    gs.execute('CREATE EDGE "a" -> "c" kind="knows"')
    
    # 3. Trigger manual rebuild to "freeze" edges into CompressedEdgeMatrix
    gs._runtime.store._rebuild_edges()
    
    # 4. Verify matrix type
    edge_matrices = gs._runtime.store._edge_matrices
    knows_matrix = edge_matrices._typed["knows"]
    
    assert isinstance(knows_matrix, CompressedEdgeMatrix)
    assert knows_matrix.nnz == 3
    
    # 5. Verify retrieval works correctly
    # RECALL should use spreading activation
    res = gs.execute('RECALL FROM "a" DEPTH 2')
    ids = [n["id"] for n in res.data]
    assert "b" in ids
    assert "c" in ids
    
    # MATCH should use BFS traversal
    res_match = gs.execute('MATCH (n)-[kind="knows"]->(m)')
    assert res_match.count == 3
    
    # 6. Verify neighbor lookups
    slot_a = gs._runtime.store.id_to_slot[gs._runtime.store.string_table.intern("a")]
    neighbors = edge_matrices.neighbors_out(slot_a, "knows")
    assert len(neighbors) == 2
    assert gs._runtime.store.id_to_slot[gs._runtime.store.string_table.intern("b")] in neighbors
    assert gs._runtime.store.id_to_slot[gs._runtime.store.string_table.intern("c")] in neighbors

def test_compressed_matrix_math():
    # Test SpMV specifically
    N = 100
    from scipy.sparse import csr_matrix
    indices = np.array([1, 2, 3], dtype=np.int32)
    indptr = np.zeros(N + 1, dtype=np.int32)
    indptr[1:] = 3 # node 0 has 3 edges, others have 0
    data = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    
    csr = csr_matrix((data, indices, indptr), shape=(N, N))
    cmat = CompressedEdgeMatrix.from_csr(csr)
    
    # Input vector: 1.0 at node 0
    vec = np.zeros(N, dtype=np.float32)
    vec[0] = 1.0
    
    # result = cmat.dot(vec)
    res = cmat.dot(vec)
    
    assert res[1] == 1.0
    assert res[2] == 1.0
    assert res[3] == 1.0
    assert np.sum(res) == 3.0
