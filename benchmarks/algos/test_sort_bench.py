
from __future__ import annotations

import numpy as np
import pytest

from supergraph.algos.sort import topk_from_column, topk_slot_order


@pytest.fixture(scope="session")
def sort_inputs_100k() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(22)
    n = 100_000
    return {
        "slots": np.arange(n, dtype=np.int32),
        "float_column": rng.random(n, dtype=np.float64),
        "int_column": rng.integers(0, 1_000_000, n, dtype=np.int64),
        "presence": rng.random(n) < 0.8,
        "values": rng.random(n, dtype=np.float64),
    }


class TestTopkSlotOrder:
    def test_100k(self, benchmark, sort_inputs_100k):
        benchmark(topk_slot_order, sort_inputs_100k["values"], True, 0, 100)


class TestTopkFromColumn:
    def test_100k_float64(self, benchmark, sort_inputs_100k):
        benchmark(
            topk_from_column,
            sort_inputs_100k["slots"],
            sort_inputs_100k["float_column"],
            sort_inputs_100k["presence"],
            "float64",
            True,
            100,
            0,
        )

    def test_100k_int64(self, benchmark, sort_inputs_100k):
        benchmark(
            topk_from_column,
            sort_inputs_100k["slots"],
            sort_inputs_100k["int_column"],
            sort_inputs_100k["presence"],
            "int64",
            True,
            100,
            0,
        )
