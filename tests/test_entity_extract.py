"""Tests for ONNX entity extraction and co-reference resolution."""
from supergraph.ingest.entity_extract import (
    extract_entities, CoReferenceResolver
)

from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models" / "tinybert-ner"


class TestEntityExtractor:
    def test_extract_person(self):
        entities = extract_entities("Caroline moved from Sweden.", model_dir=MODEL_DIR)
        persons = [e for e in entities if e.label == "PER"]
        assert len(persons) >= 1
        assert "Caroline" in [e.text for e in persons]

    def test_extract_location(self):
        entities = extract_entities("The conference was in Lisbon, Portugal.", model_dir=MODEL_DIR)
        locs = [e for e in entities if e.label == "LOC"]
        assert len(locs) >= 1
        names = [e.text for e in locs]
        assert "Lisbon" in names or "Portugal" in names

    def test_extract_organization(self):
        entities = extract_entities("She works at Google.", model_dir=MODEL_DIR)
        orgs = [e for e in entities if e.label == "ORG"]
        assert len(orgs) >= 1
        assert "Google" in [e.text for e in orgs]

    def test_empty_text(self):
        assert extract_entities("", model_dir=MODEL_DIR) == []

    def test_no_model_dir(self):
        assert extract_entities("Caroline moved.", model_dir=None) == []

    def test_high_threshold_filters_low_confidence(self):
        """Spotify subword 'ify' should be filtered at default 0.6 threshold."""
        entities = extract_entities("She works at Spotify.", model_dir=MODEL_DIR)
        assert all(e.score >= 0.6 for e in entities)
        assert not any(e.text == "ify" for e in entities)


class TestCoReferenceResolver:
    def test_resolve_she(self):
        resolver = CoReferenceResolver()
        resolver.update_context("Caroline")
        result = resolver.resolve("She started a new job.")
        assert result == ["Caroline"]

    def test_resolve_he(self):
        resolver = CoReferenceResolver()
        resolver.update_context("John")
        result = resolver.resolve("He went to the store.")
        assert result == ["John"]

    def test_no_context(self):
        resolver = CoReferenceResolver()
        result = resolver.resolve("She went home.")
        assert result == []

    def test_update_context_clears_previous(self):
        resolver = CoReferenceResolver()
        resolver.update_context("Caroline")
        resolver.update_context("Melanie")
        result = resolver.resolve("She likes hiking.")
        assert result == ["Melanie"]

    def test_non_pronoun_returns_empty(self):
        resolver = CoReferenceResolver()
        resolver.update_context("Caroline")
        result = resolver.resolve("The weather was nice.")
        assert result == []

    def test_they_pronoun(self):
        resolver = CoReferenceResolver()
        resolver.update_context("The Smiths")
        result = resolver.resolve("They moved to Sweden.")
        assert result == ["The Smiths"]


def test_slug_no_collision_on_truncation():
    from supergraph.ingest.entity_extract import slug
    a = slug("Barack Hussein Obama II the Second of That Name")
    b = slug("Barack Hussein Obama Jr the Son of the First")
    assert a != b, f"slugs collided: {a!r} == {b!r}"
    assert len(a) <= 40
    assert len(b) <= 40


def test_slug_short_unchanged():
    from supergraph.ingest.entity_extract import slug
    assert slug("Alice") == "alice"
    assert slug("Bob Smith") == "bob_smith"
