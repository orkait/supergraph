import numpy as np
from supergraph import SuperGraph
from supergraph.embedding.base import Embedder


class DeterministicEmbedder(Embedder):
    @property
    def name(self): return "deterministic"
    @property
    def dims(self): return 32

    def _hash_encode(self, texts):
        vecs = []
        for t in texts:
            seed = hash(t) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(32).astype(np.float32)
            vec /= np.linalg.norm(vec)
            vecs.append(vec)
        return np.array(vecs, dtype=np.float32)

    def encode_documents(self, texts, titles=None):
        return self._hash_encode(texts)
    def encode_queries(self, texts):
        return self._hash_encode(texts)


def test_update_embed_field_re_embeds():
    gs = SuperGraph(embedder=DeterministicEmbedder())
    gs.execute('SYS REGISTER NODE KIND "concept" REQUIRED text:string EMBED text')
    gs.execute('CREATE NODE "c1" kind = "concept" text = "old meaning"')

    slot = gs._store.id_to_slot[gs._store.string_table.intern("c1")]
    old_vec = gs._vector_store.get_vector(slot).copy()

    gs.execute('UPDATE NODE "c1" SET text = "completely different meaning"')

    new_vec = gs._vector_store.get_vector(slot)
    assert not np.allclose(old_vec, new_vec), "Vector should change after updating embed field"
    gs.close()


def test_update_non_embed_field_does_not_reembed():
    gs = SuperGraph(embedder=DeterministicEmbedder())
    gs.execute('SYS REGISTER NODE KIND "concept" REQUIRED text:string OPTIONAL score:int EMBED text')
    gs.execute('CREATE NODE "c2" kind = "concept" text = "some meaning" score = 1')

    slot = gs._store.id_to_slot[gs._store.string_table.intern("c2")]
    old_vec = gs._vector_store.get_vector(slot).copy()

    gs.execute('UPDATE NODE "c2" SET score = 99')

    new_vec = gs._vector_store.get_vector(slot)
    assert np.allclose(old_vec, new_vec), "Vector should NOT change when non-embed field updated"
    gs.close()


def test_update_without_schema_does_not_crash():
    gs = SuperGraph(embedder=DeterministicEmbedder())
    gs.execute('CREATE NODE "plain" kind = "generic" name = "test"')
    gs.execute('UPDATE NODE "plain" SET name = "updated"')
    node = gs.execute('NODE "plain"')
    assert node.data["name"] == "updated"
    gs.close()


def test_embedder_mismatch_blocks_remember(tmp_path):
    import pytest
    from supergraph import SuperGraph
    from supergraph.core.errors import SuperGraphError

    class StubEmb:
        name = "stub-A"
        dims = 4
        def encode_documents(self, texts, titles=None):
            import numpy as np
            return np.zeros((len(texts), 4), dtype=np.float32)
        def encode_queries(self, texts):
            import numpy as np
            return np.zeros((len(texts), 4), dtype=np.float32)

    class StubEmbB(StubEmb):
        name = "stub-B"

    path = tmp_path / "gs"
    gs = SuperGraph(path=str(path), embedder=StubEmb())
    try:
        gs.execute('SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text')
        gs.execute('CREATE NODE "a" kind = "doc" text = "hi"')
        gs.checkpoint()
    finally:
        gs.close()

    gs2 = SuperGraph(path=str(path), embedder=StubEmbB())
    try:
        with pytest.raises(SuperGraphError, match="SYS REEMBED"):
            gs2.execute('REMEMBER "hi" LIMIT 5')
    finally:
        gs2.close()
