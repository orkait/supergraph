"""Tests for benchmark adapter routing and retrieval tuning config output."""

from types import SimpleNamespace
from unittest.mock import Mock

from benchmarks.framework.adapters.supergraph_ import SuperGraphAdapter
from benchmarks.framework.adapters.base import QueryContext
from benchmarks.framework.runners import ratchet_recall, ratchet_test


def test_adapter_routes_categories_when_no_explicit_strategy():
    adapter = SuperGraphAdapter(config={})

    assert adapter._resolve_strategy("multi-session") == "full"
    assert adapter._resolve_strategy("temporal-reasoning") == "full"
    assert adapter._resolve_strategy("knowledge-update") == "full"
    assert adapter._resolve_strategy("single-session-user") == "full"
    assert adapter._resolve_strategy("single-session-assistant") == "full"
    assert adapter._resolve_strategy("single-session-preference") == "full"
    assert adapter._resolve_strategy("unknown") == "full"


def test_adapter_passes_temporal_anchor_to_dispatch():
    adapter = SuperGraphAdapter(config={})
    adapter._gs = SimpleNamespace()  # truthy, no internal access needed

    seen = {}

    def fake_dispatch(question: str, category: str, k: int, anchor_ms=None):
        seen["anchor_ms"] = anchor_ms
        return [], []

    adapter._dispatch = fake_dispatch  # type: ignore[method-assign]
    ctx = QueryContext(
        question="What happened on 2023-05-29?",
        category="temporal-reasoning",
        metadata={"question_date": "2023-05-30"},
    )

    adapter.query_with_context(ctx, k=5)
    assert seen["anchor_ms"] is not None


def test_adapter_ingest_done_runs_consolidation_when_enabled():
    adapter = SuperGraphAdapter(config={"enable_consolidation": True})
    execute = Mock()
    adapter._gs = SimpleNamespace(execute=execute)

    adapter.ingest_done()
    execute.assert_called_once_with("SYS CONSOLIDATE")


def test_adapter_ingest_done_skips_consolidation_by_default():
    adapter = SuperGraphAdapter(config={})
    execute = Mock()
    adapter._gs = SimpleNamespace(execute=execute)

    adapter.ingest_done()
    execute.assert_not_called()


def test_ratchet_defaults_are_locked_to_jina_small():
    assert ratchet_test.BASE_CONFIG["embedder_model"] == "jina-v5-small-retrieval"
    assert "jina-v5-small-retrieval" in ratchet_recall.run.__code__.co_consts
