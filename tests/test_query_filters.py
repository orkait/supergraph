import pytest

from supergraph.query.filters import F, compile_where


class TestLeafBuilders:
    def test_eq(self):
        assert F.eq("kind", "memory").to_dsl() == 'kind = "memory"'

    def test_ne(self):
        assert F.ne("kind", "test").to_dsl() == 'kind != "test"'

    def test_gt(self):
        assert F.gt("importance", 0.5).to_dsl() == "importance > 0.5"

    def test_gte(self):
        assert F.gte("importance", 0.5).to_dsl() == "importance >= 0.5"

    def test_lt(self):
        assert F.lt("importance", 0.5).to_dsl() == "importance < 0.5"

    def test_lte(self):
        assert F.lte("importance", 0.5).to_dsl() == "importance <= 0.5"

    def test_in(self):
        assert F.in_("topic", ["travel", "finance"]).to_dsl() == 'topic IN ("travel", "finance")'

    def test_not_in(self):
        assert F.not_in("topic", ["test"]).to_dsl() == 'NOT (topic IN ("test"))'

    def test_in_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            F.in_("topic", [])

    def test_in_non_sequence_raises(self):
        with pytest.raises(TypeError):
            F.in_("topic", "not-a-list")

    def test_is_null(self):
        assert F.is_null("deleted_at").to_dsl() == "deleted_at = NULL"

    def test_is_not_null(self):
        assert F.is_not_null("deleted_at").to_dsl() == "deleted_at != NULL"

    def test_startswith(self):
        assert F.startswith("title", "Project").to_dsl() == 'title LIKE "Project%"'

    def test_contains(self):
        assert F.contains("title", "budget").to_dsl() == 'title CONTAINS "budget"'

    def test_raw(self):
        assert F.raw("custom_fn(x) > 0").to_dsl() == "custom_fn(x) > 0"

    def test_raw_empty_raises(self):
        with pytest.raises(ValueError):
            F.raw("")

    def test_leaf_string_escape(self):
        f = F.eq("kind", 'mem"; DROP ALL; --')
        out = f.to_dsl()
        assert out.startswith('kind = "')
        assert r'\"' in out


class TestAlgebraLaws:
    def test_and_two(self):
        f = F.eq("kind", "m") & F.gt("importance", 0.5)
        assert f.to_dsl() == 'kind = "m" AND importance > 0.5'

    def test_or_two(self):
        f = F.eq("kind", "a") | F.eq("kind", "b")
        assert f.to_dsl() == '(kind = "a" OR kind = "b")'

    def test_not(self):
        f = ~F.eq("retracted", True)
        assert f.to_dsl() == "NOT (retracted = 1)"

    def test_double_negation(self):
        f = F.eq("kind", "m")
        assert (~~f) is f or (~~f).to_dsl() == f.to_dsl()

    def test_and_associativity(self):
        a, b, c = F.eq("x", 1), F.eq("y", 2), F.eq("z", 3)
        left  = (a & b) & c
        right = a & (b & c)
        assert left.to_dsl() == right.to_dsl()

    def test_or_associativity(self):
        a, b, c = F.eq("x", 1), F.eq("y", 2), F.eq("z", 3)
        assert ((a | b) | c).to_dsl() == (a | (b | c)).to_dsl()

    def test_and_with_true_is_identity(self):
        a = F.eq("x", 1)
        assert (a & F.true()).to_dsl() == a.to_dsl()
        assert (F.true() & a).to_dsl() == a.to_dsl()

    def test_or_with_false_is_identity(self):
        a = F.eq("x", 1)
        assert (a | F.false()).to_dsl() == a.to_dsl()
        assert (F.false() | a).to_dsl() == a.to_dsl()

    def test_and_with_false_collapses(self):
        a = F.eq("x", 1)
        assert (a & F.false()).to_dsl() == "false"

    def test_or_with_true_collapses(self):
        a = F.eq("x", 1)
        assert (a | F.true()).to_dsl() == "true"

    def test_and_or_precedence_parens(self):
        a, b, c, d = F.eq("a", 1), F.eq("b", 2), F.eq("c", 3), F.eq("d", 4)
        expr = a & (b | c) & d
        out = expr.to_dsl()
        assert "(b = 2 OR c = 3)" in out


class TestFromDict:
    def test_empty_dict_is_true(self):
        assert F.from_dict({}).to_dsl() == "true"

    def test_single_eq(self):
        assert F.from_dict({"kind": "memory"}).to_dsl() == 'kind = "memory"'

    def test_multiple_ands(self):
        out = F.from_dict({"kind": "memory", "importance__gt": 0.5}).to_dsl()
        assert 'kind = "memory"' in out
        assert "importance > 0.5" in out
        assert " AND " in out

    def test_op_suffix(self):
        assert F.from_dict({"importance__gte": 0.5}).to_dsl() == "importance >= 0.5"

    def test_in_op(self):
        assert F.from_dict({"topic__in": ["a", "b"]}).to_dsl() == 'topic IN ("a", "b")'

    def test_explicit_or(self):
        out = F.from_dict({"__or__": [{"kind": "a"}, {"kind": "b"}]}).to_dsl()
        assert out == '(kind = "a" OR kind = "b")'

    def test_explicit_not(self):
        out = F.from_dict({"__not__": {"retracted": True}}).to_dsl()
        assert out == "NOT (retracted = 1)"


class TestCompileWhere:
    def test_none(self):
        assert compile_where(None) is None

    def test_empty_dict(self):
        assert compile_where({}) is None

    def test_dict(self):
        assert compile_where({"kind": "memory"}) == 'kind = "memory"'

    def test_f(self):
        assert compile_where(F.eq("kind", "memory")) == 'kind = "memory"'

    def test_true_const_drops(self):
        assert compile_where(F.true()) is None

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            compile_where(42)

    def test_false_const_raises_in_where(self):
        with pytest.raises(ValueError, match="never-match"):
            compile_where(F.false())

    def test_false_via_algebra_collapse_raises(self):
        with pytest.raises(ValueError, match="never-match"):
            compile_where(F.eq("k", "m") & F.false())
