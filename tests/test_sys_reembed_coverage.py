import numpy as np
import pytest

from supergraph.core.errors import SuperGraphError
from supergraph.embedding.base import Embedder
from supergraph.store import SuperGraph

DOC1 = "alpha beta gamma delta epsilon"
DOC2 = "zeta eta theta iota kappa"


class StubEmbedder(Embedder):

    def __init__(self, name, dims=32):
        self._n = name
        self._d = dims

    @property
    def name(self):
        return self._n

    @property
    def dims(self):
        return self._d

    def _v(self, t):
        rng = np.random.default_rng(abs(hash((self._n, t))) % (2**32))
        v = rng.standard_normal(self._d).astype("float32")
        return v / (np.linalg.norm(v) + 1e-9)

    def encode_documents(self, texts, titles=None):
        return np.stack([self._v(t) for t in texts])

    def encode_queries(self, texts):
        return np.stack([self._v(t) for t in texts])


def _store(path, embedder):
    return SuperGraph(path=str(path), embedder=embedder, enable_sentence_nodes=False)


def test_reembed_reencodes_document_vectors_and_clears_dirty(tmp_path):
    p = tmp_path / "db"
    gs = _store(p, StubEmbedder("A"))
    gs.execute(f'CREATE NODE "d1" kind = "evidence" DOCUMENT "{DOC1}"')
    gs.execute(f'CREATE NODE "d2" kind = "evidence" DOCUMENT "{DOC2}"')
    gs.close()

    gs2 = _store(p, StubEmbedder("B"))
    assert gs2._embedder_dirty is True
    with pytest.raises(SuperGraphError):
        gs2.execute(f'SIMILAR TO "{DOC1}" LIMIT 2')

    r = gs2.execute("SYS REEMBED")
    assert r.data["reembedded"] == 2
    assert r.data["skipped"] == 0
    assert gs2._embedder_dirty is False

    sim = gs2.execute(f'SIMILAR TO "{DOC1}" LIMIT 2').data
    assert sim and sim[0]["id"] == "d1"
    gs2.close()


def test_reembed_keeps_dirty_when_a_vector_cannot_be_recovered(tmp_path):
    p = tmp_path / "db"
    gs = _store(p, StubEmbedder("A"))
    gs.execute(f'CREATE NODE "d1" kind = "evidence" DOCUMENT "{DOC1}"')
    vec = "[" + ", ".join("0.1" for _ in range(32)) + "]"
    gs.execute(f'CREATE NODE "v1" kind = "raw" VECTOR {vec}')
    gs.close()

    gs2 = _store(p, StubEmbedder("B"))
    r = gs2.execute("SYS REEMBED")
    assert r.data["reembedded"] >= 1
    assert r.data["skipped"] >= 1
    assert gs2._embedder_dirty is True
    gs2.close()


def test_reembed_still_covers_summary_column(tmp_path):
    p = tmp_path / "db"
    gs = _store(p, StubEmbedder("A"))
    gs.execute(f'CREATE NODE "s1" kind = "note" summary = "{DOC1}"')
    gs.close()
    gs2 = _store(p, StubEmbedder("B"))
    if gs2._embedder_dirty:
        r = gs2.execute("SYS REEMBED")
        assert r.data["reembedded"] >= 1
        assert r.data["skipped"] == 0
    gs2.close()
