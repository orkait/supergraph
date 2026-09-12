import numpy as np

from supergraph.algos.compact import (
    apply_slot_remap_to_edges,
    build_live_mask,
    slot_remap_plan,
)


def test_build_live_mask_clears_tombstones_within_range():
    node_ids = np.array([10, 11, 12, -1], dtype=np.int32)
    mask = build_live_mask(node_ids, {1, 9}, 4)
    assert np.array_equal(mask, np.array([True, False, True, False]))


def test_slot_remap_plan_marks_dead_slots_as_negative_one():
    old_to_new, new_count = slot_remap_plan(np.array([True, False, True], dtype=bool))
    assert new_count == 2
    assert np.array_equal(old_to_new, np.array([0, -1, 1], dtype=np.int32))


def test_apply_slot_remap_to_edges_drops_dead_edges():
    old_to_new = np.array([0, -1, 1], dtype=np.int32)
    result = apply_slot_remap_to_edges(
        [(0, 2, {"kind": "ok"}), (1, 2, {"kind": "drop"})],
        old_to_new,
        3,
    )
    assert result == [(0, 1, {"kind": "ok"})]
