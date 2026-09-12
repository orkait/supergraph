"""Tests for supergraph.schema - SchemaRegistry and validation."""

import pytest

from supergraph.core.errors import SchemaError
from supergraph.core.schema import SchemaRegistry


# ── 1. Empty registry ────────────────────────────────────────────────


class TestEmptyRegistry:
    def test_no_node_kinds(self):
        reg = SchemaRegistry()
        assert reg.list_node_kinds() == []

    def test_no_edge_kinds(self):
        reg = SchemaRegistry()
        assert reg.list_edge_kinds() == []

    def test_has_node_kinds_is_false(self):
        reg = SchemaRegistry()
        assert reg.has_node_kinds is False

    def test_has_edge_kinds_is_false(self):
        reg = SchemaRegistry()
        assert reg.has_edge_kinds is False


# ── 2. register_node_kind ────────────────────────────────────────────


class TestRegisterNodeKind:
    def test_stores_required_fields(self):
        reg = SchemaRegistry()
        reg.register_node_kind("Person", ["name", "age"])
        defn = reg.get_node_kind("Person")
        assert defn is not None
        assert defn.required == {"name", "age"}

    def test_stores_optional_fields(self):
        reg = SchemaRegistry()
        reg.register_node_kind("Person", ["name"], optional=["email", "bio"])
        defn = reg.get_node_kind("Person")
        assert defn is not None
        assert defn.optional == {"email", "bio"}

    def test_optional_defaults_to_empty(self):
        reg = SchemaRegistry()
        reg.register_node_kind("Person", ["name"])
        defn = reg.get_node_kind("Person")
        assert defn is not None
        assert defn.optional == set()

    def test_has_node_kinds_becomes_true(self):
        reg = SchemaRegistry()
        reg.register_node_kind("Person", ["name"])
        assert reg.has_node_kinds is True


# ── 3. register_edge_kind ────────────────────────────────────────────


class TestRegisterEdgeKind:
    def test_stores_from_and_to_kinds(self):
        reg = SchemaRegistry()
        reg.register_edge_kind("KNOWS", from_kinds=["Person"], to_kinds=["Person"])
        defn = reg.get_edge_kind("KNOWS")
        assert defn is not None
        assert defn.from_kinds == {"Person"}
        assert defn.to_kinds == {"Person"}

    def test_has_edge_kinds_becomes_true(self):
        reg = SchemaRegistry()
        reg.register_edge_kind("KNOWS", from_kinds=["Person"], to_kinds=["Person"])
        assert reg.has_edge_kinds is True


# ── 4. unregister_node_kind ──────────────────────────────────────────


class TestUnregisterNodeKind:
    def test_removes_registered_kind(self):
        reg = SchemaRegistry()
        reg.register_node_kind("Person", ["name"])
        reg.unregister_node_kind("Person")
        assert reg.get_node_kind("Person") is None
        assert reg.has_node_kinds is False

    def test_no_error_for_unknown_kind(self):
        reg = SchemaRegistry()
        reg.unregister_node_kind("Ghost")  # should not raise


# ── 5. unregister_edge_kind ──────────────────────────────────────────


class TestUnregisterEdgeKind:
    def test_removes_registered_kind(self):
        reg = SchemaRegistry()
        reg.register_edge_kind("KNOWS", from_kinds=["Person"], to_kinds=["Person"])
        reg.unregister_edge_kind("KNOWS")
        assert reg.get_edge_kind("KNOWS") is None
        assert reg.has_edge_kinds is False

    def test_no_error_for_unknown_kind(self):
        reg = SchemaRegistry()
        reg.unregister_edge_kind("Ghost")  # should not raise


# ── 6. list_node_kinds ───────────────────────────────────────────────


class TestListNodeKinds:
    def test_returns_all_registered_kinds(self):
        reg = SchemaRegistry()
        reg.register_node_kind("Person", ["name"])
        reg.register_node_kind("Company", ["title"])
        result = reg.list_node_kinds()
        assert sorted(result) == ["Company", "Person"]


# ── 7. list_edge_kinds ───────────────────────────────────────────────


class TestListEdgeKinds:
    def test_returns_all_registered_edge_kinds(self):
        reg = SchemaRegistry()
        reg.register_edge_kind("KNOWS", from_kinds=["Person"], to_kinds=["Person"])
        reg.register_edge_kind("WORKS_AT", from_kinds=["Person"], to_kinds=["Company"])
        result = reg.list_edge_kinds()
        assert sorted(result) == ["KNOWS", "WORKS_AT"]


# ── 8–9. describe_node_kind ──────────────────────────────────────────


class TestDescribeNodeKind:
    def test_returns_structured_dict(self):
        reg = SchemaRegistry()
        reg.register_node_kind("Person", ["name", "age"], optional=["email"])
        desc = reg.describe_node_kind("Person")
        assert desc == {
            "kind": "Person",
            "required": ["age", "name"],
            "optional": ["email"],
        }

    def test_returns_none_for_unknown(self):
        reg = SchemaRegistry()
        assert reg.describe_node_kind("Unknown") is None


# ── 10. describe_edge_kind ───────────────────────────────────────────


class TestDescribeEdgeKind:
    def test_returns_structured_dict(self):
        reg = SchemaRegistry()
        reg.register_edge_kind("KNOWS", from_kinds=["Person"], to_kinds=["Person", "Bot"])
        desc = reg.describe_edge_kind("KNOWS")
        assert desc == {
            "kind": "KNOWS",
            "from_kinds": ["Person"],
            "to_kinds": ["Bot", "Person"],
        }

    def test_returns_none_for_unknown(self):
        reg = SchemaRegistry()
        assert reg.describe_edge_kind("Unknown") is None


# ── 11–14. validate_node ─────────────────────────────────────────────


class TestValidateNode:
    def test_passes_when_all_required_fields_present(self):
        reg = SchemaRegistry()
        reg.register_node_kind("Person", ["name", "age"])
        reg.validate_node("Person", {"name": "Alice", "age": 30})  # no exception

    def test_raises_schema_error_when_required_fields_missing(self):
        reg = SchemaRegistry()
        reg.register_node_kind("Person", ["name", "age"])
        with pytest.raises(SchemaError, match="requires fields"):
            reg.validate_node("Person", {"name": "Alice"})

    def test_error_lists_missing_fields(self):
        reg = SchemaRegistry()
        reg.register_node_kind("Person", ["name", "age", "email"])
        with pytest.raises(SchemaError) as exc_info:
            reg.validate_node("Person", {})
        msg = str(exc_info.value)
        assert "age" in msg
        assert "email" in msg
        assert "name" in msg

    def test_passes_for_unregistered_kind(self):
        reg = SchemaRegistry()
        reg.validate_node("Alien", {})  # no exception, schema-free

    def test_extra_fields_beyond_required_and_optional_are_ok(self):
        reg = SchemaRegistry()
        reg.register_node_kind("Person", ["name"], optional=["email"])
        reg.validate_node("Person", {"name": "Bob", "email": "x", "bonus": 42})


# ── 15–18. validate_edge ─────────────────────────────────────────────


class TestValidateEdge:
    def test_passes_when_endpoint_kinds_match(self):
        reg = SchemaRegistry()
        reg.register_edge_kind("KNOWS", from_kinds=["Person"], to_kinds=["Person"])
        reg.validate_edge("KNOWS", "Person", "Person")  # no exception

    def test_raises_when_source_kind_does_not_match(self):
        reg = SchemaRegistry()
        reg.register_edge_kind("KNOWS", from_kinds=["Person"], to_kinds=["Person"])
        with pytest.raises(SchemaError, match="cannot originate from kind 'Robot'"):
            reg.validate_edge("KNOWS", "Robot", "Person")

    def test_raises_when_target_kind_does_not_match(self):
        reg = SchemaRegistry()
        reg.register_edge_kind("KNOWS", from_kinds=["Person"], to_kinds=["Person"])
        with pytest.raises(SchemaError, match="cannot target kind 'Robot'"):
            reg.validate_edge("KNOWS", "Person", "Robot")

    def test_passes_for_unregistered_edge_kind(self):
        reg = SchemaRegistry()
        reg.validate_edge("UNKNOWN_REL", "Foo", "Bar")  # no exception


# ── 19. to_dict / from_dict round-trip ───────────────────────────────


class TestRoundTrip:
    def test_preserves_all_definitions(self):
        reg = SchemaRegistry()
        reg.register_node_kind("Person", ["name", "age"], optional=["email"])
        reg.register_node_kind("Company", ["title"])
        reg.register_edge_kind("WORKS_AT", from_kinds=["Person"], to_kinds=["Company"])

        exported = reg.to_dict()
        restored = SchemaRegistry.from_dict(exported)

        assert restored.to_dict() == exported

    def test_round_trip_empty_registry(self):
        reg = SchemaRegistry()
        exported = reg.to_dict()
        restored = SchemaRegistry.from_dict(exported)
        assert restored.to_dict() == {"node_kinds": {}, "edge_kinds": {}}

    def test_from_dict_with_missing_sections(self):
        restored = SchemaRegistry.from_dict({})
        assert restored.list_node_kinds() == []
        assert restored.list_edge_kinds() == []


# ── 20. has_node_kinds / has_edge_kinds properties ───────────────────


class TestHasKindsProperties:
    def test_has_node_kinds_toggles(self):
        reg = SchemaRegistry()
        assert reg.has_node_kinds is False
        reg.register_node_kind("X", ["a"])
        assert reg.has_node_kinds is True
        reg.unregister_node_kind("X")
        assert reg.has_node_kinds is False

    def test_has_edge_kinds_toggles(self):
        reg = SchemaRegistry()
        assert reg.has_edge_kinds is False
        reg.register_edge_kind("R", from_kinds=["A"], to_kinds=["B"])
        assert reg.has_edge_kinds is True
        reg.unregister_edge_kind("R")
        assert reg.has_edge_kinds is False


# ── Typed fields ────────────────────────────────────────────────────


class TestTypedFields:
    def test_register_with_types(self):
        sr = SchemaRegistry()
        sr.register_node_kind("fn", [("name", "string"), ("line", "int")], [("score", "float")])
        defn = sr.get_node_kind("fn")
        assert "name" in defn.required
        assert defn.field_types == {"name": "string", "line": "int", "score": "float"}

    def test_validate_type_mismatch_raises(self):
        sr = SchemaRegistry()
        sr.register_node_kind("fn", [("name", "string"), ("line", "int")])
        with pytest.raises(SchemaError):
            sr.validate_node("fn", {"name": 42, "line": 1})

    def test_validate_correct_types_passes(self):
        sr = SchemaRegistry()
        sr.register_node_kind("fn", [("name", "string"), ("line", "int")])
        sr.validate_node("fn", {"name": "main", "line": 1})

    def test_untyped_fields_skip_type_check(self):
        sr = SchemaRegistry()
        sr.register_node_kind("fn", [("name", None)], [])
        sr.validate_node("fn", {"name": 42})

    def test_backward_compat_string_list(self):
        sr = SchemaRegistry()
        sr.register_node_kind("fn", ["name", "line"])
        defn = sr.get_node_kind("fn")
        assert "name" in defn.required
        assert defn.field_types == {}

    def test_describe_includes_types(self):
        sr = SchemaRegistry()
        sr.register_node_kind("fn", [("name", "string")])
        desc = sr.describe_node_kind("fn")
        assert desc["field_types"] == {"name": "string"}

    def test_serialization_preserves_types(self):
        sr = SchemaRegistry()
        sr.register_node_kind("fn", [("name", "string"), ("line", "int")])
        data = sr.to_dict()
        sr2 = SchemaRegistry.from_dict(data)
        defn = sr2.get_node_kind("fn")
        assert defn.field_types == {"name": "string", "line": "int"}
