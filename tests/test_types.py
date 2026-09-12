"""Tests for supergraph.types and supergraph.errors."""

import json
import numpy as np

from supergraph.core.types import Edge, NodeData, Result
from supergraph.core.errors import (
    BatchRollback,
    CeilingExceeded,
    CostThresholdExceeded,
    SuperGraphError,
    NodeExists,
    NodeNotFound,
    QueryError,
    SchemaError,
    VersionMismatch,
)


# ── Result ───────────────────────────────────────────────────────────

class TestResult:
    def test_to_dict(self):
        r = Result(kind="node", data={"id": "a", "name": "Alice"}, count=1, elapsed_us=42)
        d = r.to_dict()
        assert d == {
            "kind": "node",
            "data": {"id": "a", "name": "Alice"},
            "count": 1,
            "elapsed_us": 42,
        }

    def test_to_dict_default_elapsed(self):
        r = Result(kind="ok", data=None, count=0)
        assert r.elapsed_us == 0
        assert r.to_dict()["elapsed_us"] == 0

    def test_to_json_roundtrip(self):
        r = Result(kind="nodes", data=[{"id": "a"}, {"id": "b"}], count=2, elapsed_us=100)
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["kind"] == "nodes"
        assert parsed["count"] == 2
        assert len(parsed["data"]) == 2

    def test_to_json_uses_default_str(self):
        """Non-serializable objects should fall back to str()."""
        r = Result(kind="error", data=ValueError("boom"), count=1)
        j = r.to_json()
        parsed = json.loads(j)
        assert "boom" in parsed["data"]

    def test_to_json_converts_numpy_array_to_list(self):
        r = Result(kind="nodes", data=np.array([1, 2, 3], dtype=np.int32), count=3)
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["data"] == [1, 2, 3]

    def test_to_json_converts_list_of_numpy_arrays(self):
        r = Result(
            kind="nodes",
            data=[np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)],
            count=2,
        )
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["data"] == [[1, 2], [3, 4]]


# ── Edge ─────────────────────────────────────────────────────────────

class TestEdge:
    def test_basic(self):
        e = Edge(source="a", target="b", kind="KNOWS", data={"since": 2020})
        assert e.source == "a"
        assert e.target == "b"
        assert e.kind == "KNOWS"
        assert e.data == {"since": 2020}

    def test_default_data(self):
        e = Edge(source="x", target="y", kind="LINK")
        assert e.data == {}


# ── NodeData alias ───────────────────────────────────────────────────

class TestNodeData:
    def test_alias_accepts_dict(self):
        nd: NodeData = {"id": "n1", "label": "test"}
        assert isinstance(nd, dict)


# ── Errors ───────────────────────────────────────────────────────────

class TestErrors:
    def test_all_subclass_supergraph_error(self):
        errors = [
            QueryError("bad syntax"),
            NodeNotFound("n1"),
            NodeExists("n1"),
            CeilingExceeded(128, 64, "CREATE"),
            VersionMismatch(None, 2),
            SchemaError("missing field 'name'"),
            CostThresholdExceeded(5000.0, 1000.0),
            BatchRollback("CREATE n1 {}", "duplicate"),
        ]
        for err in errors:
            assert isinstance(err, SuperGraphError)
            assert isinstance(err, Exception)

    def test_query_error_full(self):
        e = QueryError("Unexpected token", position=12, query="FETCH x WHERE")
        s = str(e)
        assert "Unexpected token" in s
        assert "position 12" in s
        assert "FETCH x WHERE" in s

    def test_query_error_minimal(self):
        e = QueryError("bad query")
        s = str(e)
        assert s == "bad query"

    def test_node_not_found(self):
        e = NodeNotFound("abc")
        assert e.id == "abc"
        assert "abc" in str(e)

    def test_node_exists(self):
        e = NodeExists("abc")
        assert e.id == "abc"
        assert "abc" in str(e)

    def test_ceiling_exceeded(self):
        e = CeilingExceeded(current_mb=256, ceiling_mb=128, operation="CREATE")
        assert e.current_mb == 256
        assert e.ceiling_mb == 128
        assert "256" in str(e)
        assert "128" in str(e)
        assert "CREATE" in str(e)

    def test_version_mismatch_none(self):
        e = VersionMismatch(found=None, expected=2)
        assert e.found is None
        assert e.expected == 2
        assert "None" in str(e)
        assert "2" in str(e)

    def test_version_mismatch_string(self):
        e = VersionMismatch(found="1", expected=2)
        assert "1" in str(e)
        assert "2" in str(e)

    def test_schema_error(self):
        e = SchemaError("field 'age' must be int")
        assert "age" in str(e)

    def test_cost_threshold_exceeded(self):
        e = CostThresholdExceeded(estimated_frontier=5000.0, threshold=1000.0)
        assert e.estimated_frontier == 5000.0
        assert e.threshold == 1000.0
        assert "5000" in str(e)
        assert "1000" in str(e)

    def test_batch_rollback(self):
        e = BatchRollback(failed_statement="CREATE n1 {}", error="duplicate key")
        assert e.failed_statement == "CREATE n1 {}"
        assert "duplicate key" in str(e)
