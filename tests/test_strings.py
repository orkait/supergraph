
import pytest

from supergraph.core.strings import StringTable


class TestIntern:

    def test_sequential_ids(self):
        t = StringTable()
        assert t.intern("a") == 0
        assert t.intern("b") == 1
        assert t.intern("c") == 2

    def test_dedup_returns_same_id(self):
        t = StringTable()
        first = t.intern("hello")
        second = t.intern("hello")
        assert first == second == 0

    def test_dedup_does_not_grow_table(self):
        t = StringTable()
        t.intern("x")
        t.intern("x")
        assert len(t) == 1

    def test_interleaved_dedup(self):
        t = StringTable()
        id_a = t.intern("a")
        id_b = t.intern("b")
        assert t.intern("a") == id_a
        assert t.intern("b") == id_b
        assert t.intern("c") == 2


class TestLookup:

    def test_lookup_returns_correct_string(self):
        t = StringTable()
        t.intern("alpha")
        t.intern("beta")
        assert t.lookup(0) == "alpha"
        assert t.lookup(1) == "beta"

    def test_lookup_raises_keyerror_for_missing_id(self):
        t = StringTable()
        with pytest.raises(KeyError):
            t.lookup(0)

    def test_lookup_raises_keyerror_for_negative_id(self):
        t = StringTable()
        t.intern("a")
        with pytest.raises(KeyError):
            t.lookup(-999)

    def test_lookup_raises_keyerror_for_out_of_range(self):
        t = StringTable()
        t.intern("only")
        with pytest.raises(KeyError):
            t.lookup(1)


class TestLen:

    def test_empty(self):
        assert len(StringTable()) == 0

    def test_after_inserts(self):
        t = StringTable()
        t.intern("a")
        t.intern("b")
        assert len(t) == 2

    def test_dedup_not_counted(self):
        t = StringTable()
        t.intern("a")
        t.intern("a")
        t.intern("b")
        assert len(t) == 2


class TestContains:

    def test_present(self):
        t = StringTable()
        t.intern("yes")
        assert "yes" in t

    def test_absent(self):
        t = StringTable()
        assert "no" not in t

    def test_after_intern(self):
        t = StringTable()
        assert "x" not in t
        t.intern("x")
        assert "x" in t


class TestSerialization:

    def test_round_trip_preserves_mappings(self):
        t = StringTable()
        t.intern("foo")
        t.intern("bar")
        t.intern("baz")

        exported = t.to_list()
        t2 = StringTable.from_list(exported)

        assert len(t2) == 3
        assert t2.lookup(0) == "foo"
        assert t2.lookup(1) == "bar"
        assert t2.lookup(2) == "baz"
        assert t2.intern("foo") == 0

    def test_round_trip_empty(self):
        t = StringTable()
        t2 = StringTable.from_list(t.to_list())
        assert len(t2) == 0

    def test_to_list_returns_copy(self):
        t = StringTable()
        t.intern("a")
        exported = t.to_list()
        exported.append("b")
        assert len(t) == 1

    def test_from_list_does_not_alias_input(self):
        source = ["a", "b"]
        t = StringTable.from_list(source)
        source.append("c")
        assert len(t) == 2

    def test_from_list_allows_continued_interning(self):
        t = StringTable.from_list(["x", "y"])
        new_id = t.intern("z")
        assert new_id == 2
        assert t.lookup(2) == "z"


class TestEmptyTable:

    def test_len_zero(self):
        assert len(StringTable()) == 0

    def test_contains_is_false(self):
        assert "anything" not in StringTable()

    def test_lookup_raises(self):
        with pytest.raises(KeyError):
            StringTable().lookup(0)

    def test_to_list_empty(self):
        assert StringTable().to_list() == []


class TestLargeScale:

    def test_thousand_strings(self):
        t = StringTable()
        n = 1500
        for i in range(n):
            assert t.intern(f"str_{i}") == i

        assert len(t) == n

        for i in range(n):
            assert t.lookup(i) == f"str_{i}"

        for i in range(n):
            assert t.intern(f"str_{i}") == i

        assert len(t) == n

    def test_large_round_trip(self):
        t = StringTable()
        n = 2000
        for i in range(n):
            t.intern(f"s{i}")

        t2 = StringTable.from_list(t.to_list())
        assert len(t2) == n
        for i in range(n):
            assert t2.lookup(i) == f"s{i}"
            assert t2.intern(f"s{i}") == i
