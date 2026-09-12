import pytest

from supergraph import SuperGraph

from superclaw.memory import Memory
from superclaw.tools import Permission, SideEffect, ToolContext


@pytest.fixture
def gs():
    g = SuperGraph(embedder="none", enable_sentence_nodes=False)
    yield g
    g.close()


@pytest.fixture
def mem(gs):
    return Memory(gs)


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(workspace=tmp_path)


class TestNote:
    def test_note_stores_and_returns_id(self, mem, ctx):
        res = mem.note_tool().run({"text": "The user prefers tabs over spaces."}, ctx)
        assert res.ok
        assert res.output.startswith("mem:")

    def test_note_requires_text(self, mem, ctx):
        assert not mem.note_tool().run({"text": "  "}, ctx).ok

    def test_note_is_idempotent_on_identical_text(self, mem, ctx):
        a = mem.note_tool().run({"text": "same fact"}, ctx).output
        b = mem.note_tool().run({"text": "same fact"}, ctx).output
        assert a == b


class TestSearch:
    def test_search_finds_a_stored_note(self, mem, ctx):
        mem.note_tool().run({"text": "Deploys go through Railway with Dockerfile.cloud-cpu."}, ctx)
        mem.note_tool().run({"text": "The mascot is a purple otter."}, ctx)
        res = mem.search_tool().run({"query": "how do deploys work", "limit": 3}, ctx)
        assert res.ok
        assert "Railway" in res.output

    def test_search_empty_corpus_is_ok(self, mem, ctx):
        res = mem.search_tool().run({"query": "anything"}, ctx)
        assert res.ok
        assert "no" in res.output.lower()


class TestRecall:
    def test_recall_returns_formatted_bullets(self, mem, ctx):
        mem.note_tool().run({"text": "Kailas runs the orkait monorepo."}, ctx)
        block = mem.recall("who runs orkait", limit=3)
        assert "- Kailas runs the orkait monorepo." in block

    def test_recall_empty_is_empty_string(self, mem):
        assert mem.recall("nothing here") == ""


class TestSafety:
    def test_search_is_read_allow(self, mem):
        s = mem.search_tool().safety
        assert (s.side_effect, s.permission) == (SideEffect.READ, Permission.ALLOW)

    def test_note_is_none_allow(self, mem):
        s = mem.note_tool().safety
        assert (s.side_effect, s.permission) == (SideEffect.NONE, Permission.ALLOW)

    def test_memory_writes_to_default_namespace_not_superclaw(self, mem, gs, ctx):
        mem.note_tool().run({"text": "a durable fact"}, ctx)
        assert gs.execute("COUNT NODES").count == 1
        assert gs.execute('COUNT NODES', namespace="superclaw").count == 0
