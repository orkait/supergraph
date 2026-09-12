"""The Research facade: the lean 8-verb deep-research SDK over SuperGraph."""
import numpy as np
import pytest

from supergraph.embedding.base import Embedder
from supergraph.research import Research
from supergraph.store import SuperGraph


class StubEmbedder(Embedder):
    def __init__(self, dims=32):
        self._d = dims

    @property
    def name(self):
        return "stub"

    @property
    def dims(self):
        return self._d

    def _v(self, t):
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        v = rng.standard_normal(self._d).astype("float32")
        return v / (np.linalg.norm(v) + 1e-9)

    def encode_documents(self, texts, titles=None):
        return np.stack([self._v(t) for t in texts])

    def encode_queries(self, texts):
        return np.stack([self._v(t) for t in texts])


@pytest.fixture
def r():
    return Research(SuperGraph(embedder=StubEmbedder(), enable_sentence_nodes=False))


def test_ingest_returns_id_and_search_finds_it(r):
    eid = r.ingest("Nike signed a football sponsorship deal with the team")
    r.ingest("Heavy rainfall is expected across the coast this weekend")
    assert isinstance(eid, str)
    hits = r.search("football sponsorship")
    assert any(h.get("id") == eid for h in hits)


def test_ingest_batch_returns_list(r):
    ids = r.ingest(["alpha fact one", "beta fact two", "gamma fact three"])
    assert isinstance(ids, list) and len(ids) == 3
    assert r.execute("COUNT NODES").data == 3


def test_ingest_is_idempotent_on_same_content(r):
    a = r.ingest("the same exact content")
    b = r.ingest("the same exact content")
    assert a == b
    assert r.execute("COUNT NODES").data == 1


def test_relate_and_explore(r):
    a = r.ingest("source document about apples")
    b = r.ingest("target document about bananas")
    r.relate(a, b, kind="cites")
    neighbours = r.explore(a, depth=1)
    assert any(row.get("id") == b for row in neighbours)


def test_forget_removes_node(r):
    x = r.ingest("a temporary fact to delete")
    r.forget(x)
    assert not r.execute(f'NODES WHERE id = "{x}"').data


def test_gaps_surfaces_low_confidence_and_sparse(r):
    r.execute('CREATE NODE "weak" kind = "evidence" confidence = 0.2')
    r.execute('CREATE NODE "ent:lonely" kind = "entity" name = "Lonely" confidence = 0.9')
    g = r.gaps()
    ids = {x["id"] for x in g}
    kinds = {x["kind"] for x in g}
    assert "weak" in ids and "ent:lonely" in ids
    assert "gather" in kinds and "expand" in kinds


def test_execute_is_the_escape_hatch(r):
    assert r.execute("COUNT NODES").data == 0


def test_answer_without_reader_raises_clearly(r):
    with pytest.raises(Exception):
        r.answer("what is known about the corpus?")
