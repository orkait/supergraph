from __future__ import annotations

import sys
from pathlib import Path

import pytest

_FIXTURES = Path(__file__).parent / "fixtures" / "benchmarks"
sys.path.insert(0, str(_FIXTURES))

from benchmarks.framework.transport.llm_client import compute_f1  # noqa: E402

try:
    import snap_locomo_eval_reference as ref  # noqa: E402
except ImportError as e:  # pragma: no cover - missing deps is a real failure
    pytest.skip(f"reference not importable: {e}", allow_module_level=True)


def _ref_score(category: int, prediction: str, gold: str) -> float:
    if category == 5:
        low = prediction.lower()
        if 'no information available' in low or 'not mentioned' in low:
            return 1.0
        return 0.0
    if category == 3:
        gold = gold.split(';')[0].strip()
    if category == 1:
        return float(ref.f1(prediction, gold))
    if category in (2, 3, 4):
        return float(ref.f1_score(prediction, gold))
    raise ValueError(f"unknown category: {category}")


_FIXTURES_BY_CAT = {
    1: [
        ("Adoption agencies", "Adoption agencies"),
        ("adoption agencies, counseling", "Adoption agencies, counseling"),
        ("counseling", "Adoption agencies, counseling"),
        ("no information available", "Adoption agencies"),
    ],
    2: [
        ("2022", "2022"),
        ("7 May 2023", "7 May 2023"),
        ("trans community", "Transgender woman"),
        ("no information available", "Single"),
    ],
    3: [
        ("Psychology, counseling certification", "Psychology, counseling certification"),
        ("counseling", "Psychology, counseling certification"),
        ("20 May 2023", "The sunday before 25 May 2023"),
        ("9 June 2023", "The week before 9 June 2023"),
        ("psychology", "Psychology; alternate answer here"),
    ],
    4: [
        ("mental health", "mental health"),
        ("cancer awareness", "mental health"),
        ("no information available", "mental health"),
    ],
    5: [
        ("no information available", "Unanswerable"),
        ("not mentioned", "Unanswerable"),
        ("Chicago", "Unanswerable"),
        ("I think it was Chicago", "Unanswerable"),
    ],
}


@pytest.mark.parametrize("category,pairs", list(_FIXTURES_BY_CAT.items()))
def test_compute_f1_matches_snap_research_reference(category, pairs):
    for pred, gold in pairs:
        ours = compute_f1(pred, gold, category=category)
        theirs = _ref_score(category, pred, gold)
        assert ours == pytest.approx(theirs, abs=1e-9), (
            f"MISMATCH cat={category} pred={pred!r} gold={gold!r} "
            f"ours={ours} theirs={theirs}"
        )


def test_adversarial_abstention_phrases_match():
    for pred in ("no information available", "Not mentioned",
                 "I have no information available on that.",
                 "This was not mentioned in our chats."):
        assert compute_f1(pred, "Unanswerable", category=5) == 1.0
    for pred in ("don't remember", "unknown", "can't tell",
                 "yes", "Chicago", ""):
        assert compute_f1(pred, "Unanswerable", category=5) == 0.0


def test_temporal_semicolon_split_gold():
    pred = "psychology"
    gold = "psychology; counseling; certification"
    assert compute_f1(pred, gold, category=3) == pytest.approx(1.0, abs=1e-9)


def test_multihop_comma_split_golds():
    pred = "alice, bob"
    gold = "alice, bob, carol"
    ours = compute_f1(pred, gold, category=1)
    assert ours == pytest.approx(2/3, abs=1e-9)
