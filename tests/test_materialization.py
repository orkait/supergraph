import numpy as np

from supergraph.algos.materialization import materialize_bulk


def test_materialize_bulk_skips_reserved_fields():
    rows = materialize_bulk(
        slots=np.array([0], dtype=np.int32),
        node_ids=np.array([0], dtype=np.int32),
        node_kinds=np.array([1], dtype=np.int32),
        id_to_str=["a", "memory"],
        columns={
            "name": np.array([0], dtype=np.int32),
            "__updated_at__": np.array([123], dtype=np.int64),
        },
        presence={
            "name": np.array([True]),
            "__updated_at__": np.array([True]),
        },
        dtypes={
            "name": "int32_interned",
            "__updated_at__": "int64",
        },
    )
    assert rows == [{"id": "a", "kind": "memory", "name": "a"}]


def test_materialize_bulk_handles_partial_presence():
    rows = materialize_bulk(
        slots=np.array([0, 1], dtype=np.int32),
        node_ids=np.array([0, 1], dtype=np.int32),
        node_kinds=np.array([2, 2], dtype=np.int32),
        id_to_str=["a", "b", "memory"],
        columns={
            "name": np.array([0, 1], dtype=np.int32),
            "score": np.array([10, 20], dtype=np.int64),
        },
        presence={
            "name": np.array([True, False]),
            "score": np.array([False, True]),
        },
        dtypes={
            "name": "int32_interned",
            "score": "int64",
        },
    )
    assert rows == [
        {"id": "a", "kind": "memory", "name": "a"},
        {"id": "b", "kind": "memory", "score": 20},
    ]
