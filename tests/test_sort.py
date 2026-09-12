import numpy as np

from supergraph.algos.sort import topk_from_column, topk_slot_order


def test_topk_slot_order_descending_respects_offset_and_limit():
    values = np.array([1.0, 4.0, 3.0, 2.0], dtype=np.float64)
    order = topk_slot_order(values, descending=True, offset=1, limit=2)
    assert np.array_equal(order, np.array([2, 3]))


def test_topk_from_column_returns_none_for_interned_strings():
    slots = np.array([0, 1, 2], dtype=np.int32)
    column = np.array([1, 2, 3], dtype=np.int32)
    presence = np.array([True, True, True], dtype=bool)
    assert topk_from_column(slots, column, presence, "int32_interned", True, 10, 0) is None


def test_topk_from_column_pushes_missing_to_end_when_descending():
    slots = np.array([0, 1, 2], dtype=np.int32)
    column = np.array([10.0, 30.0, 20.0], dtype=np.float64)
    presence = np.array([True, False, True], dtype=bool)
    ordered = topk_from_column(slots, column, presence, "float64", True, None, 0)
    assert np.array_equal(ordered, np.array([2, 0, 1], dtype=np.int32))
