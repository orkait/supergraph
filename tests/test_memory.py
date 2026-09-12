
import pytest

from supergraph.core.errors import CeilingExceeded
from supergraph.core.memory import (
    BYTES_PER_EDGE,
    BYTES_PER_NODE,
    DEFAULT_CEILING_BYTES,
    check_ceiling,
    estimate,
)
from supergraph.core.store import CoreStore


class TestEstimate:
    def test_zero_nodes_zero_edges(self):
        assert estimate(0, 0) == 0

    def test_nodes_only(self):
        assert estimate(10, 0) == 10 * BYTES_PER_NODE

    def test_edges_only(self):
        assert estimate(0, 50) == 50 * BYTES_PER_EDGE

    def test_known_values(self):
        nodes, edges = 100, 500
        expected = (100 * BYTES_PER_NODE) + (500 * BYTES_PER_EDGE)
        assert estimate(nodes, edges) == expected

    def test_single_node_single_edge(self):
        assert estimate(1, 1) == BYTES_PER_NODE + BYTES_PER_EDGE

    def test_large_graph_800k_nodes_3_2m_edges(self):
        usage = estimate(800_000, 3_200_000)
        assert usage == 328_000_000
        assert usage > DEFAULT_CEILING_BYTES, (
            "800K nodes + 3.2M edges exceeds the 256MB default ceiling"
        )


class TestCheckCeiling:
    def test_under_ceiling_passes(self):
        check_ceiling(
            current_nodes=100,
            current_edges=200,
            added_nodes=10,
            added_edges=20,
        )

    def test_over_ceiling_raises(self):
        with pytest.raises(CeilingExceeded):
            check_ceiling(
                current_nodes=0,
                current_edges=0,
                added_nodes=1_000_000,
                added_edges=1_000_000,
            )

    def test_exactly_at_ceiling_does_not_raise(self):
        ceiling = estimate(100, 200)
        check_ceiling(
            current_nodes=0,
            current_edges=0,
            added_nodes=100,
            added_edges=200,
            ceiling_bytes=ceiling,
        )

    def test_one_byte_over_ceiling_raises(self):
        ceiling = estimate(100, 200)
        with pytest.raises(CeilingExceeded):
            check_ceiling(
                current_nodes=0,
                current_edges=0,
                added_nodes=100,
                added_edges=201,
                ceiling_bytes=ceiling,
            )

    def test_custom_ceiling(self):
        tiny_ceiling = 1_000
        with pytest.raises(CeilingExceeded):
            check_ceiling(
                current_nodes=0,
                current_edges=0,
                added_nodes=10,
                added_edges=0,
                ceiling_bytes=tiny_ceiling,
            )

    def test_custom_ceiling_passes_when_under(self):
        big_ceiling = 1_000_000_000
        check_ceiling(
            current_nodes=100_000,
            current_edges=500_000,
            added_nodes=50_000,
            added_edges=100_000,
            ceiling_bytes=big_ceiling,
        )

    def test_raised_exception_attributes(self):
        with pytest.raises(CeilingExceeded) as exc_info:
            check_ceiling(
                current_nodes=500_000,
                current_edges=1_000_000,
                added_nodes=500_000,
                added_edges=1_000_000,
                ceiling_bytes=100_000_000,
            )
        err = exc_info.value
        current_usage = estimate(500_000, 1_000_000)
        assert err.current_mb == current_usage // 1_000_000
        assert err.ceiling_mb == 100
        assert "500000 nodes" in err.operation
        assert "1000000 edges" in err.operation

    def test_empty_add_on_empty_graph(self):
        check_ceiling(
            current_nodes=0,
            current_edges=0,
            added_nodes=0,
            added_edges=0,
        )

    def test_custom_bytes_per_node_overrides_static(self):
        big_bpn = 10_000
        with pytest.raises(CeilingExceeded):
            check_ceiling(
                current_nodes=0,
                current_edges=0,
                added_nodes=11,
                added_edges=0,
                ceiling_bytes=100_000,
                bytes_per_node=big_bpn,
            )
        check_ceiling(
            current_nodes=0,
            current_edges=0,
            added_nodes=10,
            added_edges=0,
            ceiling_bytes=100_000,
            bytes_per_node=big_bpn,
        )


class TestSelfCalibration:
    def test_estimate_updates_after_1000_nodes(self):
        store = CoreStore(ceiling_bytes=512 * 1_000_000)
        initial = store._bytes_per_node_estimate
        for i in range(1001):
            store.put_node(f"n{i}", "test", {"v": i})
        assert store._bytes_per_node_estimate != initial, (
            "estimate should update after 1000 nodes"
        )
        assert store._bytes_per_node_estimate > 0

    def test_calibrate_at_count_advances(self):
        store = CoreStore(ceiling_bytes=512 * 1_000_000)
        for i in range(1001):
            store.put_node(f"n{i}", "test", {})
        first_threshold = store._calibrate_at_count
        assert first_threshold > 1000, "threshold should advance past 1000"

    def test_ceiling_enforced_with_calibrated_estimate(self):
        bootstrap = CoreStore(ceiling_bytes=512 * 1_000_000)
        for i in range(1001):
            bootstrap.put_node(f"n{i}", "test", {"value": i})
        calibrated_bpn = int(bootstrap._bytes_per_node_estimate)

        tight_ceiling = calibrated_bpn * 1000 - 1
        store = CoreStore(ceiling_bytes=512 * 1_000_000)
        store._bytes_per_node_estimate = float(calibrated_bpn)
        store._calibrate_at_count = 10**9
        for i in range(1000):
            store.put_node(f"n{i}", "test", {})
        with pytest.raises(CeilingExceeded):
            check_ceiling(
                store._count, 0, 1000, 0,
                ceiling_bytes=tight_ceiling,
                bytes_per_node=calibrated_bpn,
            )
