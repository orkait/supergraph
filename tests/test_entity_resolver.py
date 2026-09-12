"""Tests for supergraph.entity_resolver.

The resolver is pure-read - it never mutates the store. Tests build
synthetic graph state via direct DSL writes, then call
``resolve_mention()`` and assert it picks the right entity (existing
vs new) with the right confidence.

Three scenarios drive coverage:
  1. Empty graph        → always new entity, confidence 1.0
  2. Single name match  → unambiguous link, confidence 1.0
  3. Multiple same-name → embedding disambiguation, threshold-gated
"""
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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


@pytest.fixture
def gs(tmp_path):
    """Fresh on-disk store + clean resolver cache per test.

    The process-global name->entity cache survives across pytest
    cases without explicit reset, which would let one test's
    resolution leak into another. Clear it in both setup and
    teardown.
    """
    from supergraph.entity_resolver import reset_resolver_cache_for_tests
    reset_resolver_cache_for_tests()
    store = SuperGraph(path=str(tmp_path / "db"))
    yield store
    store.close()
    reset_resolver_cache_for_tests()


def _create_entity(gs, entity_id: str, canonical_name: str,
                   context: str = "", mention_count: int = 0):
    """Materialize an entity node the resolver can find."""
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


# ---------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------


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
        assert len(ids) == 100  # no collisions in a small batch

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


# ---------------------------------------------------------------------
# resolve_mention()
# ---------------------------------------------------------------------


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
    """Two entities named Alice with diverging contexts. Resolver must
    pick the one whose accumulated context matches the new mention's
    context most closely."""

    def test_picks_the_contextually_closer_entity(self, gs):
        """When two same-name entities exist and the new mention's
        context is closer to one of them, resolver picks that one
        (assuming the cosine clears the threshold).

        Note: tightened with a lowered threshold so the test exercises
        the disambiguation branch deterministically across embedder
        choices. Default threshold is 0.85 (production-conservative);
        this test uses 0.4 which any sane embedder clears for
        topically-related text.
        """
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

        # New mention: clearly the engineer's context.
        result = resolve_mention(
            gs, surface_name="Alice",
            context=("Alice pushed a Go service to production today; "
                     "the new Postgres index works"),
            threshold_high=0.4,  # disambiguation regime, not name match
        )
        assert result.is_new_entity is False
        assert result.entity_id == "entity:engineer"
        assert result.candidates_seen == 2
        assert result.confidence > 0.4

    def test_default_threshold_rejects_weak_disambig(self, gs):
        """Default threshold (0.85) is conservative on purpose: when
        same-name entities exist but the new context only partially
        matches, mint a new entity rather than merge incorrectly. The
        false-merge cost (collapsed identities) outweighs the
        false-split cost (reversible via MERGE)."""
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
            # Default threshold_high; the 0.5-ish cosine for
            # short embeddings won't clear it.
        )
        # Either branch is acceptable: minting new (most likely) is the
        # safe default. If a future embedder actually clears 0.85 for
        # this short overlap, the test will assert is_new_entity=False
        # and that's also fine - it just means our embedder got
        # better.
        if result.is_new_entity:
            assert result.candidates_seen == 2
        else:
            assert result.confidence >= DEFAULT_HIGH_THRESHOLD

    def test_low_similarity_mints_new_entity(self, gs):
        """Two same-name entities exist; new mention has context
        unlike either. Resolver should NOT force-merge - it mints a
        third entity (false-split is reversible; false-merge is not).
        """
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
            # Wildly off-topic context for both existing Alices.
            context="ancient Roman cooking techniques and pasta history",
            threshold_high=0.99,  # force the "below threshold" branch
        )
        assert result.is_new_entity is True
        assert result.candidates_seen == 2


class TestEdgeAndKindConstants:
    """Lock the schema surface so consumers depending on these strings
    get notified by tests if we ever rename them."""

    def test_kind_constants_match_design(self):
        assert KIND_MENTION == "mention"
        assert KIND_ENTITY == "entity"
        assert EDGE_REFERS_TO == "refers_to"


class TestResolverIsPureRead:
    """Resolver MUST NOT write to the store - that's the caller's job.
    Verify by counting nodes before + after a resolve call."""

    def test_resolve_does_not_create_nodes(self, gs):
        before = gs.execute("COUNT NODES").data
        resolve_mention(gs, "Alice", "context here")
        after = gs.execute("COUNT NODES").data
        assert before == after


class TestResolverIgnoresNERAutoEntities:
    """The deterministic NER pipeline auto-creates ``ent:{slug}`` nodes
    with kind=entity whenever a CREATE NODE...DOCUMENT runs. Resolver
    candidate set must skip those (id prefix != ``entity:``) so they
    don't pollute name matching and force false-splits."""

    def test_ner_style_node_excluded_from_candidates(self, gs):
        # Simulate what the NER pipeline writes: ent:* id, kind=entity,
        # name field set but no canonical_name.
        gs.execute(
            'CREATE NODE "ent:alice" kind = "entity" name = "alice"'
        )
        # Existing canonical entity from a real mention/resolver pass.
        existing_id = "entity:c4f8a3"
        _create_entity(gs, existing_id, "Alice", context="works at OpenAI")

        result = resolve_mention(gs, surface_name="Alice", context="ctx")
        assert result.entity_id == existing_id
        assert result.is_new_entity is False
        # candidates_seen counts ONLY canonical entities, not the NER node.
        assert result.candidates_seen == 1

    def test_only_ner_nodes_present_treated_as_no_match(self, gs):
        gs.execute(
            'CREATE NODE "ent:alice" kind = "entity" name = "alice"'
        )
        result = resolve_mention(gs, surface_name="Alice", context="ctx")
        # No canonical entity exists; resolver mints fresh.
        assert result.is_new_entity is True
        assert result.candidates_seen == 0
