from __future__ import annotations

import pytest

from supergraph import SuperGraph
from supergraph.entity_resolver import (
    DEFAULT_HIGH_THRESHOLD,
    EDGE_REFERS_TO,
    KIND_ENTITY,
    KIND_MENTION,
    ResolvedMention,
    make_entity_id,
    make_mention_id,
    normalize_name,
    resolve_mention,
)


@pytest.fixture
def gs(tmp_path):
    from supergraph.entity_resolver import reset_resolver_cache_for_tests
    reset_resolver_cache_for_tests()
    store = SuperGraph(path=str(tmp_path / "db"))
    yield store
    store.close()
    reset_resolver_cache_for_tests()


def _create_entity(gs, entity_id: str, canonical_name: str,
                   context: str = "", mention_count: int = 0):
    parts = [
        f'CREATE NODE "{entity_id}"',
        f'kind = "{KIND_ENTITY}"',
        f'canonical_name = "{canonical_name}"',
    ]
    if context:
        parts.append(f'context = "{context}"')
    parts.append(f'mention_count = {mention_count}')
    if context:
        parts.append(f'DOCUMENT "{canonical_name}. {context}"')
    else:
        parts.append(f'DOCUMENT "{canonical_name}"')
    gs.execute(" ".join(parts))


class TestNormalizeName:
    @pytest.mark.parametrize("name,expected", [
        ("Alice", "alice"),
        ("ALICE", "alice"),
        ("alice ", "alice"),
        ("Alice Smith", "alicesmith"),
        ("alice@stripe", "alicestripe"),
        ("Dr. Chen", "drchen"),
        ("", ""),
    ])
    def test_normalize_collapses_to_lowercase_alphanumeric(self, name, expected):
        assert normalize_name(name) == expected

    def test_two_surface_forms_same_normalized(self):
        assert normalize_name("Alice Smith") == normalize_name("alice  smith")
        assert normalize_name("OpenAI") == normalize_name("openai")


class TestMakeEntityId:
    def test_default_prefix(self):
        eid = make_entity_id()
        assert eid.startswith("entity:")
        assert len(eid) > len("entity:")

    def test_uniqueness_across_calls(self):
        ids = {make_entity_id() for _ in range(100)}
        assert len(ids) == 100

    def test_custom_prefix(self):
        eid = make_entity_id(prefix="ent")
        assert eid.startswith("ent:")


class TestMakeMentionId:
    def test_idempotent_for_same_args(self):
        a = make_mention_id("m1", "alice", 0)
        b = make_mention_id("m1", "alice", 0)
        assert a == b

    def test_distinct_for_different_occurrences(self):
        a = make_mention_id("m1", "alice", 0)
        b = make_mention_id("m1", "alice", 1)
        assert a != b

    def test_distinct_for_different_msgs(self):
        a = make_mention_id("m1", "alice", 0)
        b = make_mention_id("m2", "alice", 0)
        assert a != b


class TestResolveOnEmptyGraph:
    def test_always_new_entity_with_full_confidence(self, gs):
        result = resolve_mention(gs, surface_name="Alice", context="just met Alice")
        assert isinstance(result, ResolvedMention)
        assert result.is_new_entity is True
        assert result.confidence == 1.0
        assert result.candidates_seen == 0
        assert result.canonical_name == "Alice"
        assert result.entity_id.startswith("entity:")


class TestResolveSingleNameMatch:
    def test_unambiguous_link_returns_existing(self, gs):
        existing_id = "entity:abc123"
        _create_entity(gs, existing_id, "Alice",
                       context="works at OpenAI")
        result = resolve_mention(
            gs, surface_name="Alice",
            context="had coffee with Alice this morning",
        )
        assert result.is_new_entity is False
        assert result.entity_id == existing_id
        assert result.confidence == 1.0
        assert result.candidates_seen == 1

    def test_normalization_treats_alice_and_ALICE_same(self, gs):
        existing_id = "entity:abc123"
        _create_entity(gs, existing_id, "alice", context="ctx")
        result = resolve_mention(gs, surface_name="ALICE", context="ctx2")
        assert result.entity_id == existing_id
        assert result.is_new_entity is False


class TestResolveMultipleSameNameMatch:

    def test_picks_the_contextually_closer_entity(self, gs):
        _create_entity(
            gs, "entity:engineer",
            "Alice",
            context=("software engineer at Stripe building payments "
                     "infrastructure on Go and Postgres"),
        )
        _create_entity(
            gs, "entity:designer",
            "Alice",
            context=("UX designer at Figma working on prototyping "
                     "tools for product teams"),
        )

        result = resolve_mention(
            gs, surface_name="Alice",
            context=("Alice pushed a Go service to production today; "
                     "the new Postgres index works"),
            threshold_high=0.4,
        )
        assert result.is_new_entity is False
        assert result.entity_id == "entity:engineer"
        assert result.candidates_seen == 2
        assert result.confidence > 0.4

    def test_default_threshold_rejects_weak_disambig(self, gs):
        _create_entity(
            gs, "entity:engineer",
            "Alice",
            context="software engineer at Stripe",
        )
        _create_entity(
            gs, "entity:designer",
            "Alice",
            context="UX designer at Figma",
        )
        result = resolve_mention(
            gs, surface_name="Alice",
            context="Alice pushed a Go service to production",
        )
        if result.is_new_entity:
            assert result.candidates_seen == 2
        else:
            assert result.confidence >= DEFAULT_HIGH_THRESHOLD

    def test_low_similarity_mints_new_entity(self, gs):
        _create_entity(
            gs, "entity:engineer",
            "Alice",
            context="software engineer at Stripe",
        )
        _create_entity(
            gs, "entity:designer",
            "Alice",
            context="UX designer at Figma",
        )

        result = resolve_mention(
            gs, surface_name="Alice",
            context="ancient Roman cooking techniques and pasta history",
            threshold_high=0.99,
        )
        assert result.is_new_entity is True
        assert result.candidates_seen == 2


class TestEdgeAndKindConstants:

    def test_kind_constants_match_design(self):
        assert KIND_MENTION == "mention"
        assert KIND_ENTITY == "entity"
        assert EDGE_REFERS_TO == "refers_to"


class TestResolverIsPureRead:

    def test_resolve_does_not_create_nodes(self, gs):
        before = gs.execute("COUNT NODES").data
        resolve_mention(gs, "Alice", "context here")
        after = gs.execute("COUNT NODES").data
        assert before == after


class TestResolverIgnoresNERAutoEntities:

    def test_ner_style_node_excluded_from_candidates(self, gs):
        gs.execute(
            'CREATE NODE "ent:alice" kind = "entity" name = "alice"'
        )
        existing_id = "entity:c4f8a3"
        _create_entity(gs, existing_id, "Alice", context="works at OpenAI")

        result = resolve_mention(gs, surface_name="Alice", context="ctx")
        assert result.entity_id == existing_id
        assert result.is_new_entity is False
        assert result.candidates_seen == 1

    def test_only_ner_nodes_present_treated_as_no_match(self, gs):
        gs.execute(
            'CREATE NODE "ent:alice" kind = "entity" name = "alice"'
        )
        result = resolve_mention(gs, surface_name="Alice", context="ctx")
        assert result.is_new_entity is True
        assert result.candidates_seen == 0
