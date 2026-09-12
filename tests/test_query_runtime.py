"""Query object: modifiers, pipe, batch compose, immutability."""
import pytest

from supergraph import q, F


class TestImmutability:
    def test_base_unchanged_after_limit(self):
        base = q.nodes(kind="memory")
        _ = base.limit(10)
        assert "LIMIT" not in base.dsl()

    def test_base_unchanged_after_where(self):
        base = q.nodes(kind="memory")
        _ = base.where(F.gt("importance", 0.5))
        assert "importance" not in base.dsl()

    def test_base_unchanged_after_with(self):
        base = q.nodes(kind="memory")
        _ = base.with_(limit=20, order_by="x DESC")
        assert "LIMIT" not in base.dsl()
        assert "ORDER BY" not in base.dsl()


class TestModifiers:
    def test_limit(self):
        assert "LIMIT 10" in q.nodes().limit(10).dsl()

    def test_limit_negative_raises(self):
        with pytest.raises(ValueError):
            q.nodes().limit(-1)

    def test_limit_non_int_raises(self):
        with pytest.raises(ValueError):
            q.nodes().limit("10")

    def test_tokens(self):
        assert "TOKENS 4000" in q.remember("x").tokens(4000).dsl()

    def test_at(self):
        assert 'AT "2024-03"' in q.remember("x").at("2024-03").dsl()

    def test_order_by(self):
        assert "ORDER BY __event_at__ DESC" in q.nodes().order_by("__event_at__ DESC").dsl()

    def test_where_and_combines(self):
        base = q.nodes(where=F.eq("kind", "memory"))
        extended = base.where(F.gt("importance", 0.5))
        assert "kind" in extended.dsl()
        assert "importance" in extended.dsl()
        assert " AND " in extended.dsl()

    def test_where_dict_accepted(self):
        assert "importance" in q.nodes().where({"importance__gt": 0.5}).dsl()

    def test_where_none_is_noop(self):
        base = q.nodes(kind="memory")
        assert base.where(None).dsl() == base.dsl()

    def test_with_kw(self):
        base = q.nodes(kind="memory")
        extended = base.with_(limit=5, order_by="importance DESC")
        assert "LIMIT 5" in extended.dsl()
        assert "ORDER BY importance DESC" in extended.dsl()

    def test_with_unknown_kwarg_raises(self):
        with pytest.raises(TypeError, match="unexpected kwargs"):
            q.nodes().with_(bogus=1)


class TestWriteModifiers:
    def test_write_limit_refused(self):
        w = q.create_node("n1", kind="memory")
        with pytest.raises(TypeError, match="only valid on read queries"):
            w.limit(10)

    def test_write_where_refused(self):
        w = q.create_node("n1", kind="memory")
        with pytest.raises(TypeError, match="only valid on read queries"):
            w.where(F.eq("kind", "memory"))


class TestPipe:
    def test_pipe_one(self):
        def add_limit(qry, n):
            return qry.limit(n)
        out = q.nodes(kind="memory").pipe(add_limit, 10)
        assert "LIMIT 10" in out.dsl()

    def test_pipe_chain(self):
        def with_recency(qry, days):
            return qry.where(F.gte("__event_at__", f"days-{days}"))

        def for_user(qry, uid):
            return qry.where(F.eq("owner", uid))

        out = q.nodes(kind="memory").pipe(with_recency, days=7).pipe(for_user, uid="u1")
        dsl = out.dsl()
        assert "__event_at__" in dsl
        assert "owner" in dsl

    def test_pipe_requires_query_return(self):
        def bad(qry):
            return "not a query"
        with pytest.raises(TypeError, match="must return a Query"):
            q.nodes().pipe(bad)


class TestBatchCompose:
    def test_or_two_writes(self):
        batch = q.create_node("n1", kind="memory") | q.create_node("n2", kind="memory")
        dsl = batch.dsl()
        assert 'CREATE NODE "n1"' in dsl
        assert 'CREATE NODE "n2"' in dsl

    def test_or_three(self):
        batch = (
            q.create_node("n1", kind="memory")
            | q.create_node("n2", kind="memory")
            | q.create_edge("n1", "n2", kind="next")
        )
        dsl = batch.dsl()
        assert 'CREATE NODE "n1"' in dsl
        assert 'CREATE NODE "n2"' in dsl
        assert 'CREATE EDGE "n1"' in dsl

    def test_or_non_query_returns_notimplemented(self):
        # Python handles the NotImplemented dance; `q | "x"` should TypeError eventually
        with pytest.raises(TypeError):
            q.nodes() | "not a query"


class TestExecute:
    def test_execute_none_raises(self):
        with pytest.raises(TypeError, match="requires a SuperGraph"):
            q.nodes().execute(None)


class TestRepr:
    def test_repr_includes_kind(self):
        r = repr(q.nodes(kind="memory"))
        assert "[read]" in r
        assert "NODES" in r

    def test_repr_truncates_long(self):
        long_text = "x" * 200
        r = repr(q.remember(long_text))
        assert len(r) < 150
        assert "..." in r

    def test_str_is_dsl(self):
        assert str(q.node("n1")) == q.node("n1").dsl()
