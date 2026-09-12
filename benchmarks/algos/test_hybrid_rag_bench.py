import timeit
import numpy as np
from scipy.sparse import csr_matrix
import sys
import os

# Ensure we can import supergraph algos
sys.path.insert(0, os.path.abspath("."))
from supergraph.algos.spreading import spreading_activation
from supergraph.algos.fusion import weighted_remember_fusion

def bench_hybrid_rag_logic():
    N = 10000
    E = 50000
    
    # 1. Setup Mock Data
    src = np.random.randint(0, N, size=E, dtype=np.int32)
    tgt = np.random.randint(0, N, size=E, dtype=np.int32)
    data = np.ones(E, dtype=np.float32)
    mat_t = csr_matrix((data, (tgt, src)), shape=(N, N))
    
    live_mask = np.ones(N, dtype=bool)
    vec_slots = np.random.randint(0, N, size=50)
    vec_sims = np.random.rand(50)
    bm25_slots = np.random.randint(0, N, size=50)
    bm25_scores = np.random.rand(50)
    
    def current_remember():
        slot_arr = np.union1d(vec_slots, bm25_slots)
        m = len(slot_arr)
        v_sig = np.zeros(m)
        vi = np.searchsorted(slot_arr, vec_slots)
        v_sig[vi] = vec_sims
        b_sig = np.zeros(m)
        bi = np.searchsorted(slot_arr, bm25_slots)
        b_sig[bi] = bm25_scores
        final = weighted_remember_fusion(
            v_sig, b_sig, np.ones(m), np.ones(m), np.zeros(m), [0.5, 0.5, 0.0, 0.0, 0.0]
        )
        return slot_arr[np.argsort(-final)[:10]]

    seed_slots = vec_slots[:3]
    
    def proposed_expansion_single_pass():
        # Optimized spreading: one activation vector with multiple seeds
        n = len(live_mask)
        activation = np.zeros(n, dtype=np.float32)
        activation[seed_slots] = 1.0 # Multi-seed injection
        
        live_f = live_mask.astype(np.float32)
        decay_f = np.float32(0.7)
        
        for _ in range(2):
            spread = mat_t.dot(activation) * decay_f
            activation += spread
            np.multiply(activation, live_f, out=activation)
        
        activation[seed_slots] = 0.0
        return np.argsort(-activation)[:10]

    print(f"Graph Size: {N} nodes, {E} edges")
    t_curr = timeit.timeit(current_remember, number=100)
    t_prop_opt = timeit.timeit(proposed_expansion_single_pass, number=100)
    
    print(f"Current Remember: {t_curr/100*1000:.3f} ms")
    print(f"Proposed (Multi-Seed 1-pass): {t_prop_opt/100*1000:.3f} ms")
    print(f"Overhead Ratio: {t_prop_opt / t_curr:.2f}x")

if __name__ == "__main__":
    bench_hybrid_rag_logic()
