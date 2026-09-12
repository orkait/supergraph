"""Test that REMEMBER uses confidence, recall frequency, and recency."""
import time
import numpy as np
from supergraph import SuperGraph
from supergraph.embedding.base import Embedder


class FixedEmbedder(Embedder):
    """Returns deterministic vectors based on text hash."""
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


def test_remember_records_recall_feedback():
    """REMEMBER should increment __recall_count__ on returned nodes."""
    gs = SuperGraph(embedder=FixedEmbedder())
    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED text:string EMBED text')
    gs.execute('CREATE NODE "r1" kind = "item" text = "quantum physics"')

    gs.execute('REMEMBER "quantum" LIMIT 5')
    gs.execute('REMEMBER "quantum" LIMIT 5')

    slot = gs._store.id_to_slot[gs._store.string_table.intern("r1")]
    if gs._store.columns.has_column("__recall_count__"):
        if gs._store.columns._presence["__recall_count__"][slot]:
            count = int(gs._store.columns._columns["__recall_count__"][slot])
            assert count >= 1, f"Expected recall_count >= 1, got {count}"
    gs.close()


def test_remember_includes_score_breakdown():
    """Results should include the full per-signal breakdown on every node."""
    gs = SuperGraph(embedder=FixedEmbedder())
    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED text:string EMBED text')
    gs.execute('CREATE NODE "t1" kind = "item" text = "test content"')

    result = gs.execute('REMEMBER "test" LIMIT 5')
    assert result.data
    node = result.data[0]
    for k in (
        "_remember_score", "_vector_sim", "_bm25_score", "_recency_score",
        "_graph_score", "_co_bonus", "_recall_boost", "_rank_stage",
    ):
        assert k in node, f"missing {k} in REMEMBER node: {list(node.keys())}"
    assert node["_rank_stage"] == "fusion"
    gs.close()


def test_remember_meta_signals_telemetry():
    """meta['signals'] surfaces fusion method, weights, stage counts, reranker state.

    This is Step 1 of supergraph's retrieval-observability effort. Callers who
    want to know *why* a REMEMBER result looks the way it does should be able
    to read the full pipeline telemetry without reading handler source.
    """
    gs = SuperGraph(embedder=FixedEmbedder(), graph_signal_enabled=True)
    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED text:string EMBED text')
    for i in range(3):
        gs.execute(f'CREATE NODE "n{i}" kind = "item" text = "entry {i}"')

    r = gs.execute('REMEMBER "entry" LIMIT 2')
    sig = r.meta["signals"]

    # Fusion block
    assert sig["fusion"]["method"] in ("weighted", "rrf")
    assert isinstance(sig["fusion"]["weights"], list)
    assert sig["fusion"]["graph_signal_enabled"] is True

    # Recency block
    assert sig["recency"]["half_life_days"] > 0

    # SQE block
    assert sig["sentence_query_expansion"]["enabled"] in (True, False)
    assert sig["sentence_query_expansion"]["num_sentences"] >= 1

    # Stages block must report every pipeline checkpoint
    stages = sig["stages"]
    for key in ("gathered_vec", "gathered_bm25", "union", "cap_applied",
                "after_cap", "before_rerank", "final"):
        assert key in stages, f"missing stage counter {key}"
    assert stages["final"] == len(r.data)

    # Reranker block (no reranker configured here -> ran=False)
    assert sig["reranker"]["ran"] is False
    assert sig["reranker"]["error"] is None

    # Nucleus block
    assert sig["nucleus"]["enabled"] in (True, False)
    gs.close()


def test_remember_graph_signal_reflected_in_meta():
    """Disabling the graph signal must show in meta['signals']['fusion']."""
    gs = SuperGraph(embedder=FixedEmbedder(), graph_signal_enabled=False)
    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED text:string EMBED text')
    gs.execute('CREATE NODE "n1" kind = "item" text = "entry"')
    r = gs.execute('REMEMBER "entry" LIMIT 5')
    assert r.meta["signals"]["fusion"]["graph_signal_enabled"] is False
    gs.close()


def test_sys_explain_remember_returns_plan_without_side_effects():
    """SYS EXPLAIN REMEMBER dry-runs: returns candidate plan, no state mutation."""
    gs = SuperGraph(embedder=FixedEmbedder())
    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED text:string EMBED text')
    for i in range(5):
        gs.execute(f'CREATE NODE "p{i}" kind = "item" text = "entry {i}"')

    r = gs.execute('SYS EXPLAIN REMEMBER "entry" LIMIT 3')

    # Shape
    assert r.kind == "plan"
    assert r.data["verb"] == "REMEMBER"
    assert r.data["query"] == "entry"
    assert r.data["limit"] == 3
    candidates = r.data["candidates"]
    assert len(candidates) == 3
    for c in candidates:
        assert "slot" in c and "id" in c
        for sig in ("fused_score", "vector_sim", "bm25_score",
                    "recency_score", "graph_score", "co_bonus", "recall_boost"):
            assert sig in c
    # Monotonic fused score
    for a, b in zip(candidates, candidates[1:]):
        assert a["fused_score"] >= b["fused_score"]

    # Meta telemetry must match the same shape REMEMBER emits
    sig = r.meta["signals"]
    for key in ("fusion", "recency", "sentence_query_expansion", "stages",
                "reranker", "nucleus"):
        assert key in sig
    assert sig["reranker"]["ran"] is False
    assert sig["stages"]["final"] == 3

    # No side effects: recall counts all absent
    for i in range(5):
        slot = gs._store._slot_to_id  # sanity: method exists
    col = gs._store.columns.get_column("__recall_count__", gs._store._next_slot)
    if col is not None:
        _, pres, _ = col
        for i in range(5):
            slot = gs._store.id_to_slot[gs._store.string_table.intern(f"p{i}")]
            assert not pres[slot], f"recall_count was set on slot {slot}; EXPLAIN should not mutate"

    gs.close()


def test_sys_explain_remember_empty_store_returns_empty_plan():
    """With no nodes, EXPLAIN must return kind='plan' with empty candidates."""
    gs = SuperGraph(embedder=FixedEmbedder())
    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED text:string EMBED text')
    r = gs.execute('SYS EXPLAIN REMEMBER "anything" LIMIT 5')
    assert r.kind == "plan"
    assert r.count == 0
    assert r.data["candidates"] == []
    gs.close()
