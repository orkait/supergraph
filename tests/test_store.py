
import pytest

from supergraph.core.errors import CeilingExceeded, SuperGraphError, NodeExists, NodeNotFound
from supergraph.core.memory import BYTES_PER_EDGE, BYTES_PER_NODE
from supergraph.core.store import CoreStore


def _populated_store():
    s = CoreStore()
    s.put_node("alice", "person", {"age": 30, "city": "NYC"})
    s.put_node("bob", "person", {"age": 25, "city": "LA"})
    s.put_node("acme", "org", {"industry": "tech"})
    s.put_edge("alice", "bob", "knows")
    s.put_edge("alice", "acme", "works_at")
    return s


class TestPutAndGet:
    def test_put_returns_slot_index(self):
        s = CoreStore()
        slot = s.put_node("n1", "thing", {"x": 1})
        assert isinstance(slot, int)
        assert slot >= 0

    def test_get_returns_full_dict(self):
        s = CoreStore()
        s.put_node("n1", "thing", {"x": 1})
        result = s.get_node("n1")
        assert result == {"id": "n1", "kind": "thing", "x": 1}

    def test_data_is_copied_on_put(self):
        s = CoreStore()
        original = {"x": 1}
        s.put_node("n1", "thing", original)
        original["x"] = 999
        assert s.get_node("n1")["x"] == 1


class TestPutDuplicate:
    def test_raises_node_exists(self):
        s = CoreStore()
        s.put_node("n1", "thing", {})
        with pytest.raises(NodeExists):
            s.put_node("n1", "thing", {})

    def test_node_exists_carries_id(self):
        s = CoreStore()
        s.put_node("n1", "thing", {})
        with pytest.raises(NodeExists) as exc_info:
            s.put_node("n1", "thing", {})
        assert exc_info.value.id == "n1"


class TestGetMissing:
    def test_totally_unknown_id(self):
        s = CoreStore()
        assert s.get_node("nonexistent") is None

    def test_deleted_node_returns_none(self):
        s = CoreStore()
        s.put_node("n1", "thing", {})
        s.delete_node("n1")
        assert s.get_node("n1") is None


class TestUpdateNode:
    def test_update_merges_fields(self):
        s = CoreStore()
        s.put_node("n1", "thing", {"a": 1, "b": 2})
        s.update_node("n1", {"b": 20, "c": 30})
        result = s.get_node("n1")
        assert result["a"] == 1
        assert result["b"] == 20
        assert result["c"] == 30

    def test_update_preserves_id_and_kind(self):
        s = CoreStore()
        s.put_node("n1", "thing", {"a": 1})
        s.update_node("n1", {"a": 2})
        result = s.get_node("n1")
        assert result["id"] == "n1"
        assert result["kind"] == "thing"


class TestUpdateMissing:
    def test_unknown_id(self):
        s = CoreStore()
        with pytest.raises(NodeNotFound):
            s.update_node("ghost", {"x": 1})

    def test_deleted_node(self):
        s = CoreStore()
        s.put_node("n1", "thing", {})
        s.delete_node("n1")
        with pytest.raises(NodeNotFound):
            s.update_node("n1", {"x": 1})


class TestUpsert:
    def test_upsert_creates_new(self):
        s = CoreStore()
        slot = s.upsert_node("n1", "thing", {"x": 1})
        assert s.get_node("n1") == {"id": "n1", "kind": "thing", "x": 1}
        assert isinstance(slot, int)

    def test_upsert_updates_existing(self):
        s = CoreStore()
        s.put_node("n1", "thing", {"x": 1})
        s.upsert_node("n1", "thing", {"x": 2, "y": 3})
        result = s.get_node("n1")
        assert result["x"] == 2
        assert result["y"] == 3

    def test_upsert_returns_same_slot_on_update(self):
        s = CoreStore()
        slot1 = s.put_node("n1", "thing", {"x": 1})
        slot2 = s.upsert_node("n1", "thing", {"x": 2})
        assert slot1 == slot2

    def test_upsert_after_delete_creates_new(self):
        s = CoreStore()
        s.put_node("n1", "thing", {"x": 1})
        s.delete_node("n1")
        s.upsert_node("n1", "thing", {"x": 99})
        assert s.get_node("n1")["x"] == 99


class TestDeleteNode:
    def test_node_not_retrievable_after_delete(self):
        s = CoreStore()
        s.put_node("n1", "thing", {})
        s.delete_node("n1")
        assert s.get_node("n1") is None

    def test_count_decremented(self):
        s = CoreStore()
        s.put_node("n1", "thing", {})
        s.put_node("n2", "thing", {})
        assert s.node_count == 2
        s.delete_node("n1")
        assert s.node_count == 1


class TestDeleteMissing:
    def test_unknown_id(self):
        s = CoreStore()
        with pytest.raises(NodeNotFound):
            s.delete_node("ghost")

    def test_already_deleted(self):
        s = CoreStore()
        s.put_node("n1", "thing", {})
        s.delete_node("n1")
        with pytest.raises(NodeNotFound):
            s.delete_node("n1")


class TestCascadeEdgeDelete:
    def test_outgoing_edges_removed(self):
        s = _populated_store()
        s.delete_node("alice")
        assert s.get_edges_from("alice") == []
        assert s.edge_count == 0

    def test_incoming_edges_removed(self):
        s = _populated_store()
        s.delete_node("bob")
        edges_from_alice = s.get_edges_from("alice")
        targets = [e["target"] for e in edges_from_alice]
        assert "bob" not in targets
        assert "acme" in targets
        assert s.edge_count == 1


class TestEdgeCRUD:
    def test_put_and_get_edges_from(self):
        s = _populated_store()
        edges = s.get_edges_from("alice")
        assert len(edges) == 2
        targets = {e["target"] for e in edges}
        assert targets == {"bob", "acme"}

    def test_get_edges_from_with_kind_filter(self):
        s = _populated_store()
        edges = s.get_edges_from("alice", kind="knows")
        assert len(edges) == 1
        assert edges[0]["target"] == "bob"
        assert edges[0]["kind"] == "knows"

    def test_get_edges_to(self):
        s = _populated_store()
        edges = s.get_edges_to("bob")
        assert len(edges) == 1
        assert edges[0]["source"] == "alice"
        assert edges[0]["kind"] == "knows"

    def test_get_edges_to_with_kind_filter(self):
        s = _populated_store()
        edges = s.get_edges_to("acme", kind="works_at")
        assert len(edges) == 1
        assert edges[0]["source"] == "alice"

    def test_get_edges_from_missing_node(self):
        s = CoreStore()
        assert s.get_edges_from("ghost") == []

    def test_get_edges_to_missing_node(self):
        s = CoreStore()
        assert s.get_edges_to("ghost") == []


class TestEdgeMissingNode:
    def test_missing_source(self):
        s = CoreStore()
        s.put_node("bob", "person", {})
        with pytest.raises(NodeNotFound):
            s.put_edge("ghost", "bob", "knows")

    def test_missing_target(self):
        s = CoreStore()
        s.put_node("alice", "person", {})
        with pytest.raises(NodeNotFound):
            s.put_edge("alice", "ghost", "knows")

    def test_both_missing(self):
        s = CoreStore()
        with pytest.raises(NodeNotFound):
            s.put_edge("ghost1", "ghost2", "knows")

    def test_deleted_source(self):
        s = CoreStore()
        s.put_node("alice", "person", {})
        s.put_node("bob", "person", {})
        s.delete_node("alice")
        with pytest.raises(NodeNotFound):
            s.put_edge("alice", "bob", "knows")

    def test_deleted_target(self):
        s = CoreStore()
        s.put_node("alice", "person", {})
        s.put_node("bob", "person", {})
        s.delete_node("bob")
        with pytest.raises(NodeNotFound):
            s.put_edge("alice", "bob", "knows")


class TestDuplicateEdge:
    def test_exact_duplicate_raises(self):
        s = CoreStore()
        s.put_node("a", "t", {})
        s.put_node("b", "t", {})
        s.put_edge("a", "b", "rel")
        with pytest.raises(SuperGraphError, match="Duplicate edge"):
            s.put_edge("a", "b", "rel")

    def test_different_kind_allowed(self):
        s = CoreStore()
        s.put_node("a", "t", {})
        s.put_node("b", "t", {})
        s.put_edge("a", "b", "rel1")
        s.put_edge("a", "b", "rel2")
        assert s.edge_count == 2


class TestDeleteEdge:
    def test_edge_removed(self):
        s = _populated_store()
        s.delete_edge("alice", "bob", "knows")
        edges = s.get_edges_from("alice")
        targets = [e["target"] for e in edges]
        assert "bob" not in targets
        assert "acme" in targets

    def test_edge_count_decremented(self):
        s = _populated_store()
        assert s.edge_count == 2
        s.delete_edge("alice", "bob", "knows")
        assert s.edge_count == 1

    def test_delete_nonexistent_edge_no_error(self):
        s = _populated_store()
        s.delete_edge("bob", "alice", "knows")
        assert s.edge_count == 2


class TestSecondaryIndex:
    def test_build_index_from_existing(self):
        s = CoreStore()
        s.put_node("a", "person", {"city": "NYC"})
        s.put_node("b", "person", {"city": "LA"})
        s.put_node("c", "person", {"city": "NYC"})
        s.add_index("city")
        nyc_slots = s.query_by_index("city", "NYC")
        assert len(nyc_slots) == 2
        la_slots = s.query_by_index("city", "LA")
        assert len(la_slots) == 1

    def test_query_unindexed_field_returns_empty(self):
        s = CoreStore()
        s.put_node("a", "person", {"city": "NYC"})
        assert s.query_by_index("city", "NYC") == []

    def test_query_missing_value_returns_empty(self):
        s = CoreStore()
        s.put_node("a", "person", {"city": "NYC"})
        s.add_index("city")
        assert s.query_by_index("city", "MISSING") == []

    def test_nodes_without_field_not_indexed(self):
        s = CoreStore()
        s.put_node("a", "person", {"city": "NYC"})
        s.put_node("b", "person", {})
        s.add_index("city")
        all_slots = s.query_by_index("city", "NYC")
        assert len(all_slots) == 1


class TestIndexMaintenance:
    def test_index_updated_on_put(self):
        s = CoreStore()
        s.add_index("city")
        s.put_node("a", "person", {"city": "NYC"})
        assert len(s.query_by_index("city", "NYC")) == 1
        s.put_node("b", "person", {"city": "NYC"})
        assert len(s.query_by_index("city", "NYC")) == 2

    def test_index_updated_on_update(self):
        s = CoreStore()
        s.put_node("a", "person", {"city": "NYC"})
        s.add_index("city")
        assert len(s.query_by_index("city", "NYC")) == 1

        s.update_node("a", {"city": "LA"})
        assert s.query_by_index("city", "NYC") == []
        assert len(s.query_by_index("city", "LA")) == 1

    def test_index_updated_on_delete(self):
        s = CoreStore()
        s.put_node("a", "person", {"city": "NYC"})
        s.add_index("city")
        assert len(s.query_by_index("city", "NYC")) == 1

        s.delete_node("a")
        assert s.query_by_index("city", "NYC") == []


class TestArrayGrowth:
    def test_exceeds_initial_capacity(self):
        s = CoreStore()
        for i in range(1100):
            s.put_node(f"n{i}", "thing", {"i": i})
        assert s.node_count == 1100
        assert s.get_node("n0")["i"] == 0
        assert s.get_node("n1099")["i"] == 1099

    def test_capacity_doubled(self):
        s = CoreStore()
        assert s._capacity == 1024
        for i in range(1025):
            s.put_node(f"n{i}", "thing", {})
        assert s._capacity == 2048

    def test_arrays_consistent_after_growth(self):
        s = CoreStore()
        for i in range(1025):
            s.put_node(f"n{i}", "thing", {"val": i})
        for i in range(1025):
            node = s.get_node(f"n{i}")
            assert node is not None
            assert node["val"] == i


class TestTombstoneReuse:
    def test_slot_reused(self):
        s = CoreStore()
        slot0 = s.put_node("a", "thing", {})
        s.put_node("b", "thing", {})
        s.delete_node("a")
        slot2 = s.put_node("c", "thing", {})
        assert slot2 == slot0

    def test_next_slot_not_advanced_on_reuse(self):
        s = CoreStore()
        s.put_node("a", "thing", {})
        s.put_node("b", "thing", {})
        next_before = s._next_slot
        s.delete_node("a")
        s.put_node("c", "thing", {})
        assert s._next_slot == next_before


class TestCounts:
    def test_node_count_tracks_puts(self):
        s = CoreStore()
        assert s.node_count == 0
        s.put_node("a", "t", {})
        assert s.node_count == 1
        s.put_node("b", "t", {})
        assert s.node_count == 2

    def test_node_count_tracks_deletes(self):
        s = CoreStore()
        s.put_node("a", "t", {})
        s.put_node("b", "t", {})
        s.delete_node("a")
        assert s.node_count == 1

    def test_edge_count_tracks_puts(self):
        s = CoreStore()
        s.put_node("a", "t", {})
        s.put_node("b", "t", {})
        assert s.edge_count == 0
        s.put_edge("a", "b", "rel")
        assert s.edge_count == 1

    def test_edge_count_tracks_deletes(self):
        s = _populated_store()
        assert s.edge_count == 2
        s.delete_edge("alice", "bob", "knows")
        assert s.edge_count == 1

    def test_edge_count_on_cascade(self):
        s = _populated_store()
        assert s.edge_count == 2
        s.delete_node("alice")
        assert s.edge_count == 0


class TestIncrementField:
    def test_increment_existing_int(self):
        s = CoreStore()
        s.put_node("n1", "t", {"score": 10})
        s.increment_field("n1", "score", 5)
        assert s.get_node("n1")["score"] == 15

    def test_increment_existing_float(self):
        s = CoreStore()
        s.put_node("n1", "t", {"rating": 3.5})
        s.increment_field("n1", "rating", 0.5)
        assert s.get_node("n1")["rating"] == pytest.approx(4.0)

    def test_increment_missing_field_starts_at_zero(self):
        s = CoreStore()
        s.put_node("n1", "t", {})
        s.increment_field("n1", "counter", 1)
        assert s.get_node("n1")["counter"] == 1

    def test_increment_negative(self):
        s = CoreStore()
        s.put_node("n1", "t", {"score": 10})
        s.increment_field("n1", "score", -3)
        assert s.get_node("n1")["score"] == 7

    def test_increment_raises_node_not_found(self):
        s = CoreStore()
        with pytest.raises(NodeNotFound):
            s.increment_field("ghost", "x", 1)

    def test_increment_raises_type_error(self):
        s = CoreStore()
        s.put_node("n1", "t", {"name": "alice"})
        with pytest.raises(TypeError, match="not numeric"):
            s.increment_field("n1", "name", 1)


class TestGetAllNodes:
    def test_returns_all(self):
        s = _populated_store()
        nodes = s.get_all_nodes()
        assert len(nodes) == 3
        ids = {n["id"] for n in nodes}
        assert ids == {"alice", "bob", "acme"}

    def test_filter_by_kind(self):
        s = _populated_store()
        persons = s.get_all_nodes(kind="person")
        assert len(persons) == 2
        ids = {n["id"] for n in persons}
        assert ids == {"alice", "bob"}

    def test_filter_by_kind_no_match(self):
        s = _populated_store()
        assert s.get_all_nodes(kind="robot") == []

    def test_excludes_tombstoned(self):
        s = _populated_store()
        s.delete_node("alice")
        nodes = s.get_all_nodes()
        ids = {n["id"] for n in nodes}
        assert "alice" not in ids
        assert len(nodes) == 2

    def test_includes_data(self):
        s = _populated_store()
        nodes = s.get_all_nodes(kind="org")
        assert len(nodes) == 1
        assert nodes[0]["industry"] == "tech"

    def test_empty_store(self):
        s = CoreStore()
        assert s.get_all_nodes() == []


class TestMemoryCeiling:
    def test_put_node_exceeds_ceiling(self):
        ceiling = BYTES_PER_NODE * 2 - 1
        s = CoreStore(ceiling_bytes=ceiling)
        s.put_node("a", "t", {})
        with pytest.raises(CeilingExceeded):
            s.put_node("b", "t", {})

    def test_put_edge_exceeds_ceiling(self):
        ceiling = BYTES_PER_NODE * 2 + BYTES_PER_EDGE - 1
        s = CoreStore(ceiling_bytes=ceiling)
        s.put_node("a", "t", {})
        s.put_node("b", "t", {})
        with pytest.raises(CeilingExceeded):
            s.put_edge("a", "b", "rel")


class TestLiveSlotsVisibility:

    def _store_with_node(self, node_id="n1", kind="t", data=None):
        s = CoreStore()
        s.put_node(node_id, kind, data or {})
        return s

    def test_ttl_expired_node_hidden_in_get_all_nodes(self):
        s = self._store_with_node()
        slot = s.id_to_slot[s.string_table.intern("n1")]
        s.columns.set_reserved(slot, "__expires_at__", 1)
        nodes = s.get_all_nodes()
        assert nodes == [], "expired node must not appear in get_all_nodes"

    def test_ttl_future_node_visible_in_get_all_nodes(self):
        import time
        s = self._store_with_node()
        slot = s.id_to_slot[s.string_table.intern("n1")]
        future_ms = int(time.time() * 1000) + 60_000
        s.columns.set_reserved(slot, "__expires_at__", future_ms)
        nodes = s.get_all_nodes()
        assert len(nodes) == 1, "non-expired node must still appear"

    def test_retracted_node_hidden_in_get_all_nodes(self):
        s = self._store_with_node()
        slot = s.id_to_slot[s.string_table.intern("n1")]
        s.columns.set_reserved(slot, "__retracted__", 1)
        nodes = s.get_all_nodes()
        assert nodes == [], "retracted node must not appear in get_all_nodes"

    def test_ttl_expired_excluded_from_count_nodes(self):
        s = CoreStore()
        s.put_node("a", "t", {})
        s.put_node("b", "t", {})
        slot = s.id_to_slot[s.string_table.intern("b")]
        s.columns.set_reserved(slot, "__expires_at__", 1)
        assert s.count_nodes() == 1

    def test_retracted_excluded_from_count_nodes(self):
        s = CoreStore()
        s.put_node("a", "t", {})
        s.put_node("b", "t", {})
        slot = s.id_to_slot[s.string_table.intern("b")]
        s.columns.set_reserved(slot, "__retracted__", 1)
        assert s.count_nodes() == 1

    def test_ttl_expired_excluded_from_query_node_ids(self):
        s = CoreStore()
        s.put_node("a", "t", {})
        s.put_node("b", "t", {})
        slot = s.id_to_slot[s.string_table.intern("b")]
        s.columns.set_reserved(slot, "__expires_at__", 1)
        ids = s.query_node_ids()
        assert "b" not in ids
        assert "a" in ids

    def test_non_ttl_store_still_caches_live_slots(self):
        s = CoreStore()
        s.put_node("a", "t", {})
        _ = s._live_slots()
        version_before = s._live_version
        slots2 = s._live_slots()
        assert s._live_version == version_before
        assert s._live_slots_cache.get(None) is not None
        assert len(slots2) == 1

    def test_ttl_store_does_not_use_stale_cache(self):
        import time
        s = CoreStore()
        future_ms = int(time.time() * 1000) + 60_000
        s.put_node("a", "t", {})
        slot = s.id_to_slot[s.string_table.intern("a")]
        s.columns.set_reserved(slot, "__expires_at__", future_ms)
        slots1 = s._live_slots()
        assert len(slots1) == 1
        s.columns.set_reserved(slot, "__expires_at__", 1)
        slots2 = s._live_slots()
        assert len(slots2) == 0, "expired node leaked through cache"
