
from __future__ import annotations

import numpy as np
import pytest

from supergraph.algos.materialization import materialize_bulk
from supergraph.core.types import Result


def _build_case(node_count: int, batch_size: int) -> dict[str, object]:
    kind_id = node_count
    id_to_str = [f"n{i}" for i in range(node_count)]
    id_to_str.append("memory")

    node_ids = np.arange(node_count, dtype=np.int32)
    node_kinds = np.full(node_count, kind_id, dtype=np.int32)
    columns = {
        "name": np.arange(node_count, dtype=np.int32),
        "score": np.arange(node_count, dtype=np.int64),
        "ratio": np.linspace(0.0, 1.0, node_count, dtype=np.float64),
        "__updated_at__": np.full(node_count, 1_700_000_000_000, dtype=np.int64),
    }
    presence = {
        "name": np.ones(node_count, dtype=bool),
        "score": np.ones(node_count, dtype=bool),
        "ratio": np.ones(node_count, dtype=bool),
        "__updated_at__": np.ones(node_count, dtype=bool),
    }
    dtypes = {
        "name": "int32_interned",
        "score": "int64",
        "ratio": "float64",
        "__updated_at__": "int64",
    }
    slots = np.arange(batch_size, dtype=np.int32)
    rows = materialize_bulk(
        slots=slots,
        node_ids=node_ids,
        node_kinds=node_kinds,
        id_to_str=id_to_str,
        columns=columns,
        presence=presence,
        dtypes=dtypes,
    )
    result = Result(kind="nodes", data=rows, count=len(rows))
    return {
        "slots": slots,
        "node_ids": node_ids,
        "node_kinds": node_kinds,
        "id_to_str": id_to_str,
        "columns": columns,
        "presence": presence,
        "dtypes": dtypes,
        "result": result,
    }


@pytest.fixture(scope="session")
def materialization_case_1k() -> dict[str, object]:
    return _build_case(node_count=10_000, batch_size=1_000)


@pytest.fixture(scope="session")
def materialization_case_5k() -> dict[str, object]:
    return _build_case(node_count=100_000, batch_size=5_000)


@pytest.fixture(scope="session")
def materialization_case_20k() -> dict[str, object]:
    return _build_case(node_count=500_000, batch_size=20_000)


def test_materialize_bulk_1k(benchmark, materialization_case_1k):
    kwargs = {
        k: materialization_case_1k[k]
        for k in ("slots", "node_ids", "node_kinds", "id_to_str", "columns", "presence", "dtypes")
    }
    materialize_bulk(**kwargs)
    result = benchmark(materialize_bulk, **kwargs)
    assert len(result) == 1_000


def test_materialize_bulk_5k(benchmark, materialization_case_5k):
    kwargs = {
        k: materialization_case_5k[k]
        for k in ("slots", "node_ids", "node_kinds", "id_to_str", "columns", "presence", "dtypes")
    }
    materialize_bulk(**kwargs)
    result = benchmark(materialize_bulk, **kwargs)
    assert len(result) == 5_000


def test_materialize_bulk_20k(benchmark, materialization_case_20k):
    kwargs = {
        k: materialization_case_20k[k]
        for k in ("slots", "node_ids", "node_kinds", "id_to_str", "columns", "presence", "dtypes")
    }
    materialize_bulk(**kwargs)
    result = benchmark(materialize_bulk, **kwargs)
    assert len(result) == 20_000


def test_result_to_json_1k(benchmark, materialization_case_1k):
    result = materialization_case_1k["result"]
    result.to_json()
    payload = benchmark(result.to_json)
    assert payload.startswith("{")


def test_result_to_json_5k(benchmark, materialization_case_5k):
    result = materialization_case_5k["result"]
    result.to_json()
    payload = benchmark(result.to_json)
    assert payload.startswith("{")


def test_result_to_json_20k(benchmark, materialization_case_20k):
    result = materialization_case_20k["result"]
    result.to_json()
    payload = benchmark(result.to_json)
    assert payload.startswith("{")
