import numpy as np
import pytest
from supergraph.vector.store import VectorStore


class TestVectorStoreBasic:
    def test_add_and_has_vector(self):
        vs = VectorStore(dims=4, capacity=100)
        assert not vs.has_vector(0)
        vs.add(0, np.array([1.0, 0.0, 0.0, 0.0]))
        assert vs.has_vector(0)

    def test_add_wrong_dims_raises(self):
        vs = VectorStore(dims=4, capacity=100)
        with pytest.raises(ValueError, match="Expected 4 dims"):
            vs.add(0, np.array([1.0, 0.0]))

    def test_remove(self):
        vs = VectorStore(dims=4, capacity=100)
        vs.add(0, np.array([1.0, 0.0, 0.0, 0.0]))
        vs.remove(0)
        assert not vs.has_vector(0)

    def test_count(self):
        vs = VectorStore(dims=4, capacity=100)
        assert vs.count() == 0
        vs.add(0, np.array([1.0, 0.0, 0.0, 0.0]))
        vs.add(1, np.array([0.0, 1.0, 0.0, 0.0]))
        assert vs.count() == 2

    def test_dims_property(self):
        vs = VectorStore(dims=256)
        assert vs.dims == 256

    def test_memory_bytes(self):
        vs = VectorStore(dims=256, capacity=1000)
        for i in range(10):
            vs.add(i, np.random.randn(256).astype(np.float32))
        assert vs.memory_bytes > 0


class TestVectorStoreSearch:
    def test_search_finds_nearest(self):
        vs = VectorStore(dims=4, capacity=100)
        vs.add(0, np.array([1.0, 0.0, 0.0, 0.0]))
        vs.add(1, np.array([0.9, 0.1, 0.0, 0.0]))
        vs.add(2, np.array([0.0, 0.0, 1.0, 0.0]))
        slots, dists = vs.search(np.array([1.0, 0.0, 0.0, 0.0]), k=2)
        assert len(slots) == 2
        assert 0 in slots
        assert 1 in slots

    def test_search_with_mask(self):
        vs = VectorStore(dims=4, capacity=100)
        vs.add(0, np.array([1.0, 0.0, 0.0, 0.0]))
        vs.add(1, np.array([0.9, 0.1, 0.0, 0.0]))
        vs.add(2, np.array([0.0, 0.0, 1.0, 0.0]))
        mask = np.zeros(100, dtype=bool)
        mask[1] = True
        mask[2] = True
        slots, _ = vs.search(np.array([1.0, 0.0, 0.0, 0.0]), k=2, mask=mask)
        assert 0 not in slots
        assert 1 in slots

    def test_search_empty_index(self):
        vs = VectorStore(dims=4, capacity=100)
        slots, dists = vs.search(np.array([1.0, 0.0, 0.0, 0.0]), k=5)
        assert len(slots) == 0

    def test_search_k_larger_than_count(self):
        vs = VectorStore(dims=4, capacity=100)
        vs.add(0, np.array([1.0, 0.0, 0.0, 0.0]))
        slots, dists = vs.search(np.array([1.0, 0.0, 0.0, 0.0]), k=100)
        assert len(slots) == 1


class TestVectorStorePersistence:
    def test_save_and_load(self):
        vs = VectorStore(dims=4, capacity=100)
        vs.add(0, np.array([1.0, 0.0, 0.0, 0.0]))
        vs.add(1, np.array([0.0, 1.0, 0.0, 0.0]))
        data = vs.save()
        assert len(data) > 0

        vs2 = VectorStore(dims=4, capacity=100)
        vs2.load(data)
        slots, _ = vs2.search(np.array([1.0, 0.0, 0.0, 0.0]), k=2)
        assert 0 in slots

    def test_save_empty_index(self):
        vs = VectorStore(dims=4, capacity=100)
        data = vs.save()
        assert isinstance(data, bytes)


class TestVectorStoreGrow:
    def test_grow_extends_capacity(self):
        vs = VectorStore(dims=4, capacity=2)
        vs.add(0, np.array([1.0, 0.0, 0.0, 0.0]))
        vs.add(1, np.array([0.0, 1.0, 0.0, 0.0]))
        vs.add(5, np.array([0.0, 0.0, 1.0, 0.0]))
        assert vs.has_vector(5)
        assert vs.count() == 3
