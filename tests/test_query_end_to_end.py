from __future__ import annotations

import tempfile

import pytest

from supergraph import SuperGraph, q, F


@pytest.fixture
def gs():
    with tempfile.TemporaryDirectory() as td:
        g = SuperGraph(path=f"{td}/db")
        yield g
        g.close()


@pytest.fixture
def gs_mem():
    g = SuperGraph(path=None)
    yield g
    g.close()


class TestReadsEndToEnd:
    def test_create_then_node_lookup(self, gs_mem):
        q.create_node("m1", kind="memory", document="Paris").execute(gs_mem)
        r = q.node("m1").execute(gs_mem)
        assert r.kind == "node"

    def test_nodes_with_filter(self, gs_mem):
        q.create_node("m1", kind="memory", document="a").execute(gs_mem)
        q.create_node("m2", kind="fact",   document="b").execute(gs_mem)
        r = q.nodes(kind="memory", limit=10).execute(gs_mem)
        assert r.count == 1

    def test_remember_returns_results(self, gs_mem):
        q.create_node("m1", kind="memory",
                      document="Paris is capital of France").execute(gs_mem)
        q.create_node("m2", kind="memory",
                      document="Rome is capital of Italy").execute(gs_mem)
        r = q.remember("French capital", limit=5).execute(gs_mem)
        assert r.count >= 1

    def test_lexical_matches_keyword(self, gs_mem):
        q.create_node("m1", kind="memory",
                      document="Eiffel Tower in Paris").execute(gs_mem)
        r = q.lexical("Eiffel", limit=5).execute(gs_mem)
        assert r.count >= 1

    def test_count_nodes(self, gs_mem):
        for i in range(5):
            q.create_node(f"m{i}", kind="memory", document=f"d{i}").execute(gs_mem)
        r = q.count_nodes(where=F.eq("kind", "memory")).execute(gs_mem)
        assert r.data == 5


class TestWritesEndToEnd:
    def test_update_node(self, gs_mem):
        q.create_node("m1", kind="memory", document="x", topic="travel").execute(gs_mem)
        q.update_node("m1", topic="finance").execute(gs_mem)
        r = q.node("m1").execute(gs_mem)
        assert r.data["topic"] == "finance"

    def test_create_edge_and_recall(self, gs_mem):
        q.create_node("a", kind="memory", document="A").execute(gs_mem)
        q.create_node("b", kind="memory", document="B").execute(gs_mem)
        q.create_edge("a", "b", kind="next").execute(gs_mem)
        r = q.recall("a", depth=2, limit=10).execute(gs_mem)
        ids = [n.get("id") for n in r.data]
        assert "b" in ids

    def test_delete_nodes_where(self, gs_mem):
        q.create_node("m1", kind="memory", document="x").execute(gs_mem)
        q.create_node("m2", kind="memory", document="y").execute(gs_mem)
        q.delete_nodes(where=F.eq("kind", "memory")).execute(gs_mem)
        r = q.count_nodes(where=F.eq("kind", "memory")).execute(gs_mem)
        assert r.data == 0


class TestBatchEndToEnd:
    def test_batch_executes_all(self, gs_mem):
        batch = q.batch(
            q.create_node("n1", kind="memory", document="a"),
            q.create_node("n2", kind="memory", document="b"),
            q.create_edge("n1", "n2", kind="next"),
        )
        batch.execute(gs_mem)
        r = q.count_nodes(where=F.eq("kind", "memory")).execute(gs_mem)
        assert r.data == 2

    def test_batch_var_assign(self, gs_mem):
        batch = q.batch(
            q.var("x", q.create_node("n1", kind="memory", document="a")),
            q.var("y", q.create_node("n2", kind="memory", document="b")),
            q.create_edge("$x", "$y", kind="next"),
        )
        batch.execute(gs_mem)
        r = q.recall("n1", depth=2, limit=10).execute(gs_mem)
        ids = [n.get("id") for n in r.data]
        assert "n2" in ids


class TestFilterEndToEnd:
    def test_where_dict_equivalent(self, gs_mem):
        q.create_node("m1", kind="memory", importance=0.9, document="x").execute(gs_mem)
        q.create_node("m2", kind="memory", importance=0.3, document="y").execute(gs_mem)
        r1 = q.nodes(where={"importance__gt": 0.5}).execute(gs_mem)
        r2 = q.nodes(where=F.gt("importance", 0.5)).execute(gs_mem)
        assert r1.count == r2.count == 1

    def test_compound_and(self, gs_mem):
        q.create_node("m1", kind="memory", importance=0.9, document="x").execute(gs_mem)
        q.create_node("m2", kind="memory", importance=0.3, document="y").execute(gs_mem)
        r = q.nodes(where=F.eq("kind", "memory") & F.gt("importance", 0.5)).execute(gs_mem)
        assert r.count == 1

    def test_or_and_not(self, gs_mem):
        q.create_node("a", kind="m", importance=0.9, document="x").execute(gs_mem)
        q.create_node("b", kind="f", importance=0.1, document="y").execute(gs_mem)
        r = q.nodes(where=F.eq("kind", "m") | F.eq("kind", "f")).execute(gs_mem)
        assert r.count == 2


class TestSysEndToEnd:
    def test_status_returns_node_count(self, gs_mem):
        for i in range(3):
            q.create_node(f"m{i}", kind="memory", document=f"d{i}").execute(gs_mem)
        r = q.sys.status().execute(gs_mem)
        assert isinstance(r.data, dict)
        assert r.data.get("nodes") == 3

    def test_kinds_lists_registered(self, gs_mem):
        q.sys.register_node_kind("post", required={"title": "string"}).execute(gs_mem)
        r = q.sys.kinds().execute(gs_mem)
        names = []
        for item in r.data:
            if isinstance(item, dict):
                names.append(item.get("kind") or item.get("name"))
            else:
                names.append(str(item))
        assert "post" in names

    def test_register_node_kind_then_create(self, gs_mem):
        q.sys.register_node_kind(
            "post",
            required={"title": "string"},
            optional={"views": "int"},
        ).execute(gs_mem)
        q.create_node("p1", kind="post", title="Hello").execute(gs_mem)
        r = q.nodes(kind="post").execute(gs_mem)
        assert r.count == 1


class TestModifierChain:
    def test_base_unchanged_after_chain_execute(self, gs_mem):
        for i in range(3):
            q.create_node(f"m{i}", kind="memory", document=f"d{i}").execute(gs_mem)
        base = q.nodes(kind="memory")
        ten = base.limit(10).execute(gs_mem)
        one = base.limit(1).execute(gs_mem)
        assert ten.count == 3
        assert one.count == 1


class TestImmutability:
    def test_same_query_can_execute_twice(self, gs_mem):
        q.create_node("m1", kind="memory", document="x").execute(gs_mem)
        query = q.nodes(kind="memory", limit=10)
        r1 = query.execute(gs_mem)
        r2 = query.execute(gs_mem)
        assert r1.count == r2.count == 1
