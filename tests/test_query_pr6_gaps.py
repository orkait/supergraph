"""PR 6: close the remaining sub-grammar gaps.

Covers:
  - F.like / F.similar_score / F.indegree / F.outdegree
  - field_ref dot notation (field.subfield)
  - CREATE NODE AUTO
  - q.var() + variable-aware CREATE EDGE
  - Time namespace (NOW / TODAY / YESTERDAY / NOW() - Nu)
"""
import pytest

from supergraph import q, F, Time
from supergraph.dsl.parser import parse


def _roundtrip(query_obj):
    dsl = query_obj.dsl()
    try:
        parse(dsl)
    except Exception as e:
        pytest.fail(f"parser rejected {dsl!r}: {e}")
    return dsl


class TestLike:
    def test_basic(self):
        dsl = _roundtrip(q.nodes(where=F.like("title", "proj%")))
        assert 'title LIKE "proj%"' in dsl

    def test_injection_safe(self):
        out = F.like("title", 'pj"; DROP; --').to_dsl()
        assert r'\"' in out


class TestContainsStartswith:
    def test_contains(self):
        dsl = _roundtrip(q.nodes(where=F.contains("title", "budget")))
        assert 'title CONTAINS "budget"' in dsl

    def test_startswith_as_like(self):
        """F.startswith emits LIKE "x%" since grammar has no STARTSWITH."""
        dsl = _roundtrip(q.nodes(where=F.startswith("title", "Proj")))
        assert 'title LIKE "Proj%"' in dsl


class TestIn:
    def test_in(self):
        dsl = _roundtrip(q.nodes(where=F.in_("topic", ["travel", "finance"])))
        assert 'topic IN ("travel", "finance")' in dsl

    def test_not_in_wrapped(self):
        dsl = _roundtrip(q.nodes(where=F.not_in("topic", ["test"])))
        # Emits as NOT (... IN (...))
        assert "NOT" in dsl
        assert "IN" in dsl


class TestNull:
    def test_is_null(self):
        dsl = _roundtrip(q.nodes(where=F.is_null("deleted_at")))
        assert "deleted_at = NULL" in dsl

    def test_is_not_null(self):
        dsl = _roundtrip(q.nodes(where=F.is_not_null("deleted_at")))
        assert "deleted_at != NULL" in dsl


class TestSimilarScore:
    def test_basic(self):
        dsl = _roundtrip(q.nodes(where=F.similar_score("content", "European history", gt=0.75)))
        assert 'SIMILAR(content, "European history") > 0.75' in dsl

    def test_compose_with_and(self):
        pred = F.eq("kind", "memory") & F.similar_score("content", "Paris", gt=0.5)
        dsl = _roundtrip(q.nodes(where=pred))
        assert "SIMILAR" in dsl
        assert "kind" in dsl


class TestDegree:
    def test_indegree(self):
        dsl = _roundtrip(q.nodes(where=F.indegree(">", 10)))
        assert "INDEGREE > 10" in dsl

    def test_outdegree_with_field(self):
        dsl = _roundtrip(q.nodes(where=F.outdegree(">=", 5, field="kind")))
        assert "OUTDEGREE kind >= 5" in dsl

    def test_degree_invalid_op(self):
        with pytest.raises(ValueError):
            q.nodes(where=F.indegree("~=", 10)).dsl()


class TestFieldRefDot:
    def test_dot_notation(self):
        dsl = _roundtrip(q.nodes(where=F.eq("parent.kind", "memory")))
        assert 'parent.kind = "memory"' in dsl

    def test_more_than_one_dot_rejected(self):
        # F.eq accepts, compile rejects via dsl_field_ref
        with pytest.raises(ValueError, match="at most one dot"):
            F.eq("a.b.c", 1).to_dsl()


class TestCreateNodeAuto:
    def test_basic(self):
        dsl = _roundtrip(q.create_node_auto(kind="memory"))
        assert dsl == 'CREATE NODE AUTO kind = "memory"'

    def test_full(self):
        dsl = _roundtrip(q.create_node_auto(kind="memory", topic="travel",
                                             event_at="2024-03-15", document="x"))
        assert "CREATE NODE AUTO" in dsl
        assert 'kind = "memory"' in dsl
        assert 'DOCUMENT "x"' in dsl


class TestVarAssign:
    def test_var_in_batch(self):
        batch = q.batch(
            q.var("x", q.create_node("n1", kind="memory", document="a")),
            q.var("y", q.create_node("n2", kind="memory", document="b")),
            q.create_edge("$x", "$y", kind="next"),
        )
        dsl = _roundtrip(batch)
        assert "$x = CREATE NODE" in dsl
        assert "$y = CREATE NODE" in dsl
        assert 'CREATE EDGE $x -> $y kind = "next"' in dsl

    def test_var_without_dollar_ok(self):
        """q.var('x', ...) should work same as q.var('$x', ...)."""
        out = q.var("x", q.create_node("n", kind="m")).dsl()
        assert out.startswith("$x = ")

    def test_var_rejects_non_write(self):
        with pytest.raises(ValueError, match="must be a write"):
            q.var("x", q.nodes(kind="memory"))


class TestTimeExpr:
    def test_now(self):
        assert Time.now().to_dsl() == "NOW()"

    def test_today(self):
        assert Time.today().to_dsl() == "TODAY"

    def test_yesterday(self):
        assert Time.yesterday().to_dsl() == "YESTERDAY"

    def test_now_minus(self):
        assert Time.now_minus(7, "d").to_dsl() == "NOW() - 7d"

    def test_now_minus_invalid_unit(self):
        with pytest.raises(ValueError):
            Time.now_minus(7, "y")

    def test_now_minus_negative(self):
        with pytest.raises(ValueError):
            Time.now_minus(-1, "d")

    def test_time_in_where(self):
        dsl = _roundtrip(q.nodes(where=F.gte("__event_at__", Time.now_minus(30, "d"))))
        assert "__event_at__ >= NOW() - 30d" in dsl

    def test_time_in_create_node_event_at(self):
        dsl = q.create_node("m", kind="memory", event_at=Time.today()).dsl()
        # parser doesn't accept EVENT_AT with TODAY token in grammar since
        # event_clause: "EVENT_AT" value; value accepts time_expr -> TODAY
        parse(dsl)
        assert "EVENT_AT TODAY" in dsl
