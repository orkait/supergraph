import timeit
import numpy as np
from scipy.sparse import csr_matrix

def bench_query_planning_logic():
    N = 1000000
    E = 5000000
    
    # 1. Setup Mock Data
    src = np.random.randint(0, N, size=E, dtype=np.int32)
    tgt = np.random.randint(0, N, size=E, dtype=np.int32)
    data = np.ones(E, dtype=np.float32)
    mat = csr_matrix((data, (src, tgt)), shape=(N, N))
    mat_t = mat.T.tocsr()
    
    a_slots = np.random.choice(N, 100000, replace=False)
    b_slots = np.random.choice(N, 10, replace=False)
    
    def exec_forward():
        activation = np.zeros(N, dtype=np.float32)
        activation[a_slots] = 1.0
        next_act = mat.dot(activation)
        res_mask = next_act > 0
        final = np.zeros(N, dtype=bool)
        final[b_slots] = True
        return np.where(res_mask & final)[0]

    def exec_backward():
        activation = np.zeros(N, dtype=np.float32)
        activation[b_slots] = 1.0
        next_act = mat_t.dot(activation)
        res_mask = next_act > 0
        final = np.zeros(N, dtype=bool)
        final[a_slots] = True
        return np.where(res_mask & final)[0]

    print(f"Graph Size: {N} nodes, {E} edges")
    print(f"Candidate A: 100,000 nodes, Candidate B: 10 nodes")
    
    t_fwd = timeit.timeit(exec_forward, number=10)
    t_bwd = timeit.timeit(exec_backward, number=10)
    
    print(f"Forward Execution (from A): {t_fwd/10*1000:.3f} ms")
    print(f"Backward Execution (from B): {t_bwd/10*1000:.3f} ms")
    print(f"Speedup Ratio: {t_fwd / t_bwd:.2f}x")

if __name__ == "__main__":
    bench_query_planning_logic()
