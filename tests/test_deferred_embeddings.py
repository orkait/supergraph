"""Tests for the deferred_embeddings context manager.

The deferred embedding path is the critical perf fix for transformer embedders
(EmbeddingGemma, Harrier, bge-*, etc.) where per-call inference overhead
dominates. These tests verify both correctness (same vectors as the immediate
path) and that batching actually happens.
"""
import numpy as np
from supergraph import SuperGraph
from supergraph.embedding.base import Embedder


class CountingEmbedder(Embedder):
    """Embedder that records each encode_documents call's batch size.

    Useful for verifying that deferred mode actually batches calls rather than
    falling back to one-at-a-time encoding.
    """
    def __init__(self):
        self.calls: list[int] = []  # batch size of each encode_documents call

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
    """Within deferred_embeddings, N CREATE NODEs should trigger 1 batched embed call,
    not N calls. Each node produces 2 embeddings (1 sentence + 1 parent) for short text."""
    emb = CountingEmbedder()
    gs = SuperGraph(embedder=emb)
    gs.execute('SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text')

    with gs.deferred_embeddings(batch_size=64):
        for i in range(10):
            gs.execute(f'CREATE NODE "d{i}" kind = "doc" text = "doc number {i}"')

    # Each CREATE NODE produces 2 embeddings (1 sentence + 1 parent).
    # With batch_size=64 and 20 total embeddings, exactly one flush at context exit.
    assert emb.calls == [20], f"expected one batched call of size 20, got {emb.calls}"
    gs.close()


def test_deferred_mode_auto_flushes_when_batch_size_reached():
    """Deferred mode should auto-flush when the pending queue hits batch_size."""
    emb = CountingEmbedder()
    gs = SuperGraph(embedder=emb)
    gs.execute('SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text')

    with gs.deferred_embeddings(batch_size=4):
        for i in range(10):
            gs.execute(f'CREATE NODE "d{i}" kind = "doc" text = "doc {i}"')

    # 10 inserts, each produces 2 embeddings (sentence + parent) = 20 total.
    # With batch_size=4:
    #   - after 2nd insert (4 embeddings): auto-flush (call 1, size 4)
    #   - after 4th insert (4 more): auto-flush (call 2, size 4)
    #   - after 6th insert (4 more): auto-flush (call 3, size 4)
    #   - after 8th insert (4 more): auto-flush (call 4, size 4)
    #   - context exit: final 4 embeddings (call 5, size 4)
    assert emb.calls == [4, 4, 4, 4, 4], f"expected [4, 4, 4, 4, 4], got {emb.calls}"
    gs.close()


def test_deferred_mode_retrieval_returns_correct_sentences():
    """After deferred ingestion, SIMILAR TO queries should return the right sentence nodes."""
    gs = SuperGraph(embedder=CountingEmbedder())
    gs.execute('SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text')
    with gs.deferred_embeddings(batch_size=8):
        for i in range(6):
            gs.execute(f'CREATE NODE "d{i}" kind = "doc" text = "unique content number {i}"')
    # Vectors are at sentence level; SIMILAR TO returns sentence nodes.
    result = gs.execute('SIMILAR TO "unique content number 3" LIMIT 3')
    ids = [n["id"] for n in result.data]
    # Sentence node d3:s0 should be in results (d3's first sentence).
    assert "d3:s0" in ids or "d3" in ids, f"expected d3:s0 in top-3 after deferred ingest, got {ids}"
    gs.close()


# REMOVED: test_deferred_mode_produces_same_vectors_as_immediate
# Parent nodes no longer have direct vectors after pipeline refactor.
# Vectors live at sentence level; this test tested obsolete parent-vector behavior.


# REMOVED: test_deferred_mode_with_document_clause_no_double_embed
# Pipeline refactor adds sentence-level embeddings. Parent is embedded once,
# sentence is embedded once. The original "double embed" concern (EMBED + DOCUMENT
# both embedding the parent) is no longer the relevant behavior.


def test_deferred_mode_restores_prior_state_on_exception():
    """If an exception occurs inside deferred_embeddings, the defer flag must be reset."""
    gs = SuperGraph(embedder=CountingEmbedder())
    gs.execute('SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text')

    assert gs._executor._defer_embeddings is False
    try:
        with gs.deferred_embeddings(batch_size=4):
            gs.execute('CREATE NODE "ok1" kind = "doc" text = "fine"')
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    # Defer flag must be reset even after exception
    assert gs._executor._defer_embeddings is False
    # And subsequent non-deferred CREATE NODE should still work
    gs.execute('CREATE NODE "ok2" kind = "doc" text = "after exception"')
    gs.close()
