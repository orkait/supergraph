import timeit
import numpy as np
from scipy.sparse import csr_matrix

class MockPCSR:
    def __init__(self, n, capacity_edges):
        self.n = n
        self.capacity = capacity_edges
        self.edges_src = np.full(capacity_edges, -1, dtype=np.int32)
        self.edges_tgt = np.full(capacity_edges, -1, dtype=np.int32)
        self.row_ptrs = np.zeros(n + 1, dtype=np.int32)
        self.write_pos = 0

    def add_edge(self, u, v):
        self.edges_src[self.write_pos] = u
        self.edges_tgt[self.write_pos] = v
        self.write_pos += 1

def bench_pcsr_vs_csr():
    N = 1000000
    E_initial = 5000000
    E_new = 10000
    
    src = np.random.randint(0, N, size=E_initial, dtype=np.int32)
    tgt = np.random.randint(0, N, size=E_initial, dtype=np.int32)
    data = np.ones(E_initial, dtype=np.float32)
    
    def csr_rebuild():
        new_src = np.random.randint(0, N, size=E_new, dtype=np.int32)
        new_tgt = np.random.randint(0, N, size=E_new, dtype=np.int32)
        all_src = np.concatenate([src, new_src])
        all_tgt = np.concatenate([tgt, new_tgt])
        all_data = np.ones(len(all_src), dtype=np.float32)
        return csr_matrix((all_data, (all_src, all_tgt)), shape=(N, N))

    pcsr = MockPCSR(N, E_initial + E_new * 10)
    def pcsr_inplace():
        for _ in range(E_new):
            pcsr.add_edge(np.random.randint(0, N), np.random.randint(0, N))

    print(f"Graph Size: {N} nodes, {E_initial} edges. Adding {E_new} edges.")
    t_csr = timeit.timeit(csr_rebuild, number=5)
    t_pcsr = timeit.timeit(pcsr_inplace, number=5)
    
    print(f"CSR Rebuild: {t_csr/5*1000:.3f} ms")
    print(f"PCSR In-place: {t_pcsr/5*1000:.3f} ms")
    print(f"Write Throughput Speedup: {t_csr / t_pcsr:.2f}x")

if __name__ == "__main__":
    bench_pcsr_vs_csr()
