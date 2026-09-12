import timeit
import numpy as np
from usearch.index import Index

def bench_vector_masking():
    N = 100000
    dims = 128
    k = 10
    
    index = Index(ndim=dims, metric="cos", dtype="f32")
    vectors = np.random.rand(N, dims).astype(np.float32)
    index.add(np.arange(N), vectors)
    
    mask = np.zeros(N, dtype=bool)
    mask[np.random.choice(N, 1000, replace=False)] = True
    
    query = np.random.rand(dims).astype(np.float32)
    
    def search_static_oversample():
        oversample = k * 5
        results = index.search(query, oversample)
        valid = []
        for key, dist in zip(results.keys, results.distances):
            if mask[int(key)]:
                valid.append(key)
                if len(valid) >= k:
                    break
        return valid

    def search_adaptive_oversample():
        current_oversample = k * 5
        while True:
            results = index.search(query, current_oversample)
            valid = []
            for key, dist in zip(results.keys, results.distances):
                if mask[int(key)]:
                    valid.append(key)
                    if len(valid) >= k:
                        break
            if len(valid) >= k or current_oversample >= 10000:
                break
            current_oversample *= 2
        return valid

    print(f"Index Size: {N}, Mask Density: 1%")
    t_static = timeit.timeit(search_static_oversample, number=100)
    t_adaptive = timeit.timeit(search_adaptive_oversample, number=100)
    
    res_static = search_static_oversample()
    res_adaptive = search_adaptive_oversample()
    
    print(f"Static (5x): {t_static/100*1000:.3f} ms (Found {len(res_static)} nodes)")
    print(f"Adaptive: {t_adaptive/100*1000:.3f} ms (Found {len(res_adaptive)} nodes)")
    print(f"Recall Gain: {len(res_adaptive) - len(res_static)} nodes")

if __name__ == "__main__":
    bench_vector_masking()
