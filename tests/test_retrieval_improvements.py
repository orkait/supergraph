

import numpy as np

from supergraph import SuperGraph
from supergraph.algos.fusion import rrf_remember_fusion
from supergraph.embedding.base import Embedder


class FixedEmbedder(Embedder):
    @property
    def name(self): return "fixed"
    @property
    def dims(self): return 32

    def _encode(self, texts):
        vecs = []
        for t in texts:
            seed = hash(t) % (2**31)
            rng = np.random.RandomState(seed)
            v = rng.randn(32).astype(np.float32)
            v /= np.linalg.norm(v)
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

    def encode_documents(self, texts, titles=None):
        return self._encode(texts)
    def encode_queries(self, texts):
        return self._encode(texts)


class KeywordEmbedder(Embedder):

    @property
    def name(self): return "keyword"

    @property
    def dims(self): return 4

    def _vec(self, text: str) -> np.ndarray:
        t = text.lower()
        if "quantum" in t:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if "pasta" in t:
            return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    def encode_documents(self, texts, titles=None):
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32)

    def encode_queries(self, texts):
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32)


def _make_gs(**kwargs):
    gs = SuperGraph(embedder=FixedEmbedder(), **kwargs)
    gs.execute('SYS REGISTER NODE KIND "fact" REQUIRED claim:string EMBED claim')
    gs.execute('SYS REGISTER NODE KIND "decision" REQUIRED claim:string EMBED claim')
    gs.execute('SYS REGISTER NODE KIND "entity" REQUIRED claim:string EMBED claim')
    gs.execute('SYS REGISTER NODE KIND "lesson" REQUIRED claim:string EMBED claim')
    gs.execute('SYS REGISTER NODE KIND "session" REQUIRED claim:string EMBED claim')
    return gs


def _event_at_for(gs: SuperGraph, node_id: str) -> int | None:
    slot = gs._store.id_to_slot[gs._store.string_table.intern(node_id)]
    col = gs._store.columns.get_column("__event_at__", gs._store._next_slot)
    if col is None:
        return None
    data, present, _ = col
    return int(data[slot]) if present[slot] else None


class TestRRFFusion:
    def test_rrf_returns_correct_shape(self):
        n = 100
        sig1 = np.zeros(n)
        sig2 = np.zeros(n)
        candidates = np.array([5, 10, 20, 50])
        sig1[candidates] = [0.9, 0.7, 0.3, 0.1]
        sig2[candidates] = [0.1, 0.3, 0.7, 0.9]

        fused = rrf_remember_fusion(sig1, sig2, candidate_slots=candidates, k_rrf=60.0)
        assert fused.shape == (n,)
        assert fused.dtype == np.float64

    def test_rrf_nonzero_only_at_candidates(self):
        n = 50
        sig = np.zeros(n)
        candidates = np.array([3, 7, 12])
        sig[candidates] = [0.5, 0.8, 0.3]

        fused = rrf_remember_fusion(sig, candidate_slots=candidates, k_rrf=60.0)
        non_cand = np.setdiff1d(np.arange(n), candidates)
        assert np.all(fused[non_cand] == 0.0)
        assert np.all(fused[candidates] > 0.0)

    def test_rrf_ranking_matches_dominant_signal(self):
        n = 20
        candidates = np.array([1, 2, 3, 4])
        sig1 = np.zeros(n)
        sig1[candidates] = [1.0, 0.5, 0.2, 0.1]
        sig2 = np.zeros(n)
        sig2[candidates] = [0.9, 0.4, 0.3, 0.2]

        fused = rrf_remember_fusion(sig1, sig2, candidate_slots=candidates, k_rrf=60.0)
        assert fused[1] > fused[2] > fused[3] > fused[4]

    def test_rrf_consensus_beats_single_signal(self):
        n = 20
        candidates = np.array([1, 2])
        sig1 = np.zeros(n); sig1[1] = 1.0; sig1[2] = 0.5
        sig2 = np.zeros(n); sig2[2] = 0.8
        sig3 = np.zeros(n); sig3[2] = 0.7

        fused = rrf_remember_fusion(sig1, sig2, sig3, candidate_slots=candidates, k_rrf=60.0)
        assert fused[2] > fused[1]

    def test_rrf_empty_candidates(self):
        n = 10
        sig = np.zeros(n)
        fused = rrf_remember_fusion(sig, candidate_slots=np.array([], dtype=np.int64))
        assert np.all(fused == 0.0)

    def test_rrf_k_parameter_affects_scores(self):
        n = 10
        candidates = np.array([0, 1])
        sig = np.zeros(n); sig[0] = 1.0; sig[1] = 0.5

        fused_low_k = rrf_remember_fusion(sig, candidate_slots=candidates, k_rrf=1.0)
        fused_high_k = rrf_remember_fusion(sig, candidate_slots=candidates, k_rrf=100.0)
        assert fused_low_k[0] > fused_high_k[0]


class TestConfigWiring:
    def test_tuned_defaults_promoted(self):
        gs = SuperGraph(embedder=FixedEmbedder())
        assert gs._executor._fusion_method == "weighted"
        assert gs._executor._nucleus_expansion is False
        assert gs._executor._search_oversample == 16
        gs.close()

    def test_fusion_method_wired(self):
        gs = SuperGraph(embedder=FixedEmbedder(), fusion_method="weighted")
        assert gs._executor._fusion_method == "weighted"
        gs.close()

    def test_rrf_k_wired(self):
        gs = SuperGraph(embedder=FixedEmbedder(), rrf_k=30.0)
        assert gs._executor._rrf_k == 30.0
        gs.close()

    def test_nucleus_expansion_wired(self):
        gs = SuperGraph(embedder=FixedEmbedder(), nucleus_expansion=True)
        assert gs._executor._nucleus_expansion is True
        gs.close()

    def test_nucleus_off_by_default(self):
        gs = SuperGraph(embedder=FixedEmbedder())
        assert gs._executor._nucleus_expansion is False
        gs.close()
