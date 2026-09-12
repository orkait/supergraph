"""Tests for Embedder interface and Model2Vec embedder."""

import numpy as np
import pytest
from supergraph.embedding.model2vec_embedder import Model2VecEmbedder
from supergraph.embedding.postprocess import l2_normalize, truncate_dims

pytestmark = pytest.mark.needs_embedder


class TestModel2VecEmbedder:
    @pytest.fixture(scope="class")
    def embedder(self):
        return Model2VecEmbedder()

    def test_name(self, embedder):
        assert embedder.name == "model2vec"

    def test_dims(self, embedder):
        assert isinstance(embedder.dims, int)
        assert embedder.dims > 0

    def test_encode_queries_shape(self, embedder):
        vecs = embedder.encode_queries(["hello world"])
        assert vecs.shape == (1, embedder.dims)
        assert vecs.dtype == np.float32

    def test_encode_documents_shape(self, embedder):
        vecs = embedder.encode_documents(["hello", "world"])
        assert vecs.shape == (2, embedder.dims)
        assert vecs.dtype == np.float32

    def test_similar_texts_closer(self, embedder):
        v1 = embedder.encode_queries(["Paris France"])[0]
        v2 = embedder.encode_queries(["France Paris"])[0]
        v3 = embedder.encode_queries(["quantum physics equations"])[0]
        sim_close = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        sim_far = float(np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3)))
        assert sim_close > sim_far

    def test_encode_empty_list(self, embedder):
        vecs = embedder.encode_queries([])
        assert vecs.shape[0] == 0


class TestPostprocess:
    def test_l2_normalize(self):
        v = np.array([[3.0, 4.0]], dtype=np.float32)
        normed = l2_normalize(v)
        assert abs(np.linalg.norm(normed[0]) - 1.0) < 1e-6

    def test_l2_normalize_zero_vector(self):
        v = np.array([[0.0, 0.0]], dtype=np.float32)
        normed = l2_normalize(v)
        assert not np.any(np.isnan(normed))

    def test_truncate_dims(self):
        v = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        t = truncate_dims(v, 2)
        assert t.shape == (1, 2)
        # Should be re-normalized
        assert abs(np.linalg.norm(t[0]) - 1.0) < 1e-6
