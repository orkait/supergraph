import numpy as np
from supergraph import SuperGraph
from supergraph.embedding.base import Embedder


class CountingEmbedder(Embedder):
    def __init__(self):
        self.calls: list[int] = []

    @property
    def name(self) -> str:
        return "counting"

    @property
    def dims(self) -> int:
        return 16

    def _hash_encode(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            seed = hash(t) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(16).astype(np.float32)
            vec /= np.linalg.norm(vec)
            vecs.append(vec)
        return np.array(vecs, dtype=np.float32)

    def encode_documents(self, texts, titles=None):
        self.calls.append(len(texts))
        return self._hash_encode(texts)

    def encode_queries(self, texts):
        return self._hash_encode(texts)


def test_deferred_mode_batches_create_node():
    emb = CountingEmbedder()
    gs = SuperGraph(embedder=emb)
    gs.execute('SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text')

    with gs.deferred_embeddings(batch_size=64):
        for i in range(10):
            gs.execute(f'CREATE NODE "d{i}" kind = "doc" text = "doc number {i}"')

    assert emb.calls == [20], f"expected one batched call of size 20, got {emb.calls}"
    gs.close()


def test_deferred_mode_auto_flushes_when_batch_size_reached():
    emb = CountingEmbedder()
    gs = SuperGraph(embedder=emb)
    gs.execute('SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text')

    with gs.deferred_embeddings(batch_size=4):
        for i in range(10):
            gs.execute(f'CREATE NODE "d{i}" kind = "doc" text = "doc {i}"')

    assert emb.calls == [4, 4, 4, 4, 4], f"expected [4, 4, 4, 4, 4], got {emb.calls}"
    gs.close()


def test_deferred_mode_retrieval_returns_correct_sentences():
    gs = SuperGraph(embedder=CountingEmbedder())
    gs.execute('SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text')
    with gs.deferred_embeddings(batch_size=8):
        for i in range(6):
            gs.execute(f'CREATE NODE "d{i}" kind = "doc" text = "unique content number {i}"')
    result = gs.execute('SIMILAR TO "unique content number 3" LIMIT 3')
    ids = [n["id"] for n in result.data]
    assert "d3:s0" in ids or "d3" in ids, f"expected d3:s0 in top-3 after deferred ingest, got {ids}"
    gs.close()


def test_deferred_mode_restores_prior_state_on_exception():
    gs = SuperGraph(embedder=CountingEmbedder())
    gs.execute('SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text')

    assert gs._executor._defer_embeddings is False
    try:
        with gs.deferred_embeddings(batch_size=4):
            gs.execute('CREATE NODE "ok1" kind = "doc" text = "fine"')
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert gs._executor._defer_embeddings is False
    gs.execute('CREATE NODE "ok2" kind = "doc" text = "after exception"')
    gs.close()
