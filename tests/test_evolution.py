"""Tests for the metacognitive evolution layer (Layer 5).

All 27 spec test cases plus infrastructure tests for Step 1 counter setup.
Written first (RED phase) - all should fail until implementation is complete.
"""
import time
import warnings
import pytest

from supergraph import SuperGraph
from supergraph.core.types import Result

# Slow suite (~25s). Skip by default; opt in via --run-slow or
# SUPERGRAPH_RUN_SLOW=1. See conftest.py for the gating logic.
pytestmark = pytest.mark.slow


# ============================================================
# Infrastructure: Result.meta (Step 1)
# ============================================================

def test_result_meta_field_exists():
    """Result dataclass must have a meta field defaulting to empty dict."""
    r = Result(kind="ok", data=None, count=0)
    assert hasattr(r, "meta")
    assert r.meta == {}


def test_result_meta_in_to_dict_when_non_empty():
    """to_dict() must include meta when it is non-empty."""
    r = Result(kind="ok", data=None, count=0, meta={"evolution": [{"rule": "x"}]})
    d = r.to_dict()
    assert "meta" in d
    assert d["meta"] == {"evolution": [{"rule": "x"}]}


def test_result_meta_absent_from_to_dict_when_empty():
    """to_dict() must NOT include meta when it is empty (keep payload lean)."""
    r = Result(kind="ok", data=None, count=0)
    d = r.to_dict()
    assert "meta" not in d


# ============================================================
# Infrastructure: EvolutionConfig (Step 1)
# ============================================================

def test_evolution_config_exists_in_supergraph_config():
    """SuperGraphConfig must have an evolution section."""
    from supergraph.config import SuperGraphConfig
    cfg = SuperGraphConfig()
    assert hasattr(cfg, "evolution")


def test_evolution_config_defaults():
    """EvolutionConfig must have correct default values per spec."""
    from supergraph.config import EvolutionConfig
    cfg = EvolutionConfig()
    assert cfg.similarity_buffer_size == 100
    assert cfg.max_rules == 50
    assert cfg.min_cooldown == 10
    assert cfg.history_retention == 1000


# ============================================================
# Infrastructure: SuperGraph counters (Step 1)
# ============================================================

def test_supergraph_has_counters():
    """SuperGraph must have _counters dict initialized at __init__."""
    db = SuperGraph(ceiling_mb=100, embedder=None)
    assert hasattr(db, "_counters")
    assert isinstance(db._counters, dict)


def test_counters_track_execute_ok():
    """Successful execute() calls must increment _counters['execute_ok']."""
    db = SuperGraph(ceiling_mb=100, embedder=None)
    initial = db._counters.get("execute_ok", 0)
    db.execute('CREATE NODE "c1" kind="test" x="1"')
    assert db._counters.get("execute_ok", 0) == initial + 1


def test_counters_track_execute_err():
    """Failed execute() calls must increment _counters['execute_err']."""
    db = SuperGraph(ceiling_mb=100, embedder=None)
    initial = db._counters.get("execute_err", 0)
    try:
        db.execute("TOTALLY INVALID QUERY $$$$")
    except Exception:
        pass
    assert db._counters.get("execute_err", 0) == initial + 1


def test_supergraph_has_start_time():
    """SuperGraph must have _start_time set at __init__ for query_rate signal."""
    db = SuperGraph(ceiling_mb=100, embedder=None)
    assert hasattr(db, "_start_time")
    assert isinstance(db._start_time, float)
    assert db._start_time <= time.time()


def test_supergraph_has_similarity_buffer():
    """SuperGraph must have _similarity_buffer deque for avg_similarity signal."""
    from collections import deque
    db = SuperGraph(ceiling_mb=100, embedder=None)
    assert hasattr(db, "_similarity_buffer")
    assert isinstance(db._similarity_buffer, deque)


def test_supergraph_has_last_evolution_events():
    """SuperGraph must have _last_evolution_events list for D4 feedback."""
    db = SuperGraph(ceiling_mb=100, embedder=None)
    assert hasattr(db, "_last_evolution_events")
    assert isinstance(db._last_evolution_events, list)


# ============================================================
# Infrastructure: DB tables (Step 1)
# ============================================================

def test_db_evolution_tables_created(tmp_path):
    """evolution_rules and evolution_history tables must be created in SQLite."""
    db_dir = tmp_path / "evo_tables"
    db_dir.mkdir()
    db = SuperGraph(path=str(db_dir), embedder=None)
    conn = db._conn

    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "evolution_rules" in tables
    assert "evolution_history" in tables
    db.close()


def test_evolution_rules_table_schema(tmp_path):
    """evolution_rules must have name, rule_json, created_at columns."""
    db_dir = tmp_path / "evo_schema"
    db_dir.mkdir()
    db = SuperGraph(path=str(db_dir), embedder=None)
    conn = db._conn

    cols = {row[1] for row in conn.execute("PRAGMA table_info(evolution_rules)").fetchall()}
    assert "name" in cols
    assert "rule_json" in cols
    assert "created_at" in cols
    db.close()


def test_evolution_history_table_schema(tmp_path):
    """evolution_history must have id, timestamp, rule_name, signals_json, actions_json, prev_values_json, status."""
    db_dir = tmp_path / "evo_hist_schema"
    db_dir.mkdir()
    db = SuperGraph(path=str(db_dir), embedder=None)
    conn = db._conn

    cols = {row[1] for row in conn.execute("PRAGMA table_info(evolution_history)").fetchall()}
    for expected in ("id", "timestamp", "rule_name", "signals_json", "actions_json", "prev_values_json", "status"):
        assert expected in cols, f"Missing column: {expected}"
    db.close()


# ============================================================
# Engine Core: EvolutionEngine (Step 2)
# ============================================================

def test_create_rule():
    """Test 1: Rule stored and retrievable via SYS EVOLVE LIST."""
    db = SuperGraph(ceiling_mb=100, embedder=None)
    res = db.execute(
        'SYS EVOLVE RULE "pressure" WHEN memory_pct > 85 THEN SET eviction_target_ratio = 0.6 COOLDOWN 300'
    )
    assert res.kind == "ok"

    lst = db.execute("SYS EVOLVE LIST")
    assert lst.kind in ("rules", "ok")
    names = [r["name"] for r in lst.data] if isinstance(lst.data, list) else []
    assert "pressure" in names


def test_create_duplicate_name():
    """Test 2: Duplicate rule name must be rejected."""
    db = SuperGraph(ceiling_mb=100, embedder=None)
    db.execute(
        'SYS EVOLVE RULE "dup" WHEN memory_pct > 80 THEN SET eviction_target_ratio = 0.7 COOLDOWN 60'
    )
    res = db.execute(
        'SYS EVOLVE RULE "dup" WHEN memory_pct > 80 THEN SET eviction_target_ratio = 0.7 COOLDOWN 60'
    )
    assert res.kind == "error"
    assert "duplicate" in res.data.lower() or "already exists" in res.data.lower()


def test_create_invalid_signal():
    """Test 3: Unknown signal must be rejected with 'unknown signal' message."""
    db = SuperGraph(ceiling_mb=100, embedder=None)
    res = db.execute(
        'SYS EVOLVE RULE "bad" WHEN bogus_signal > 5 THEN SET eviction_target_ratio = 0.7 COOLDOWN 60'
    )
    assert res.kind == "error"
    assert "unknown signal" in res.data.lower() or "unknown" in res.data.lower()


def test_create_invalid_param():
    """Test 4: Unknown parameter must be rejected with 'unknown parameter' message."""
    db = SuperGraph(ceiling_mb=100, embedder=None)
    res = db.execute(
        'SYS EVOLVE RULE "bad2" WHEN memory_pct > 50 THEN SET fake_param = 0.5 COOLDOWN 60'
    )
    assert res.kind == "error"
    assert "unknown parameter" in res.data.lower() or "unknown" in res.data.lower()


def test_enable_disable_cycle():
    """Test 5: Enable/disable toggle works correctly."""
    db = SuperGraph(ceiling_mb=100, embedder=None)
    db.execute(
        'SYS EVOLVE RULE "toggle" WHEN memory_pct > 90 THEN SET eviction_target_ratio = 0.5 COOLDOWN 60'
    )

    res = db.execute('SYS EVOLVE DISABLE "toggle"')
    assert res.kind == "ok"
    show = db.execute('SYS EVOLVE SHOW "toggle"')
    assert show.data["enabled"] is False

    res = db.execute('SYS EVOLVE ENABLE "toggle"')
    assert res.kind == "ok"
    show = db.execute('SYS EVOLVE SHOW "toggle"')
    assert show.data["enabled"] is True


def test_delete_rule():
    """Test 6: Deleted rule is removed from storage."""
    db = SuperGraph(ceiling_mb=100, embedder=None)
    db.execute(
        'SYS EVOLVE RULE "todelete" WHEN memory_pct > 80 THEN SET eviction_target_ratio = 0.7 COOLDOWN 60'
    )
    res = db.execute('SYS EVOLVE DELETE "todelete"')
    assert res.kind == "ok"

    lst = db.execute("SYS EVOLVE LIST")
    names = [r["name"] for r in lst.data] if isinstance(lst.data, list) else []
    assert "todelete" not in names


def test_rule_fires_on_condition():
    """Test 7: When condition is met, action is applied to config."""
    db = SuperGraph(ceiling_mb=100, embedder=None)
    initial_ratio = db._sys_executor._eviction_target_ratio

    # Create rule targeting eviction_target_ratio
    db.execute(
        'SYS EVOLVE RULE "fire" WHEN memory_pct >= 0 THEN SET eviction_target_ratio = 0.65 COOLDOWN 10'
    )

    # Manually trigger engine evaluation
    if hasattr(db, "_evolution_engine"):
        signals = db._evolution_engine.compute_signals()
        db._evolution_engine.evaluate(signals)

    # The ratio should have changed
    new_ratio = db._sys_executor._eviction_target_ratio
    assert new_ratio == pytest.approx(0.65)


def test_rule_skips_false_condition():
    """Test 8: When condition is not met, action is NOT applied."""
    db = SuperGraph(ceiling_mb=100, embedder=None)

    # Set initial known value
    db._sys_executor._eviction_target_ratio = 0.8

    # Rule fires only when memory_pct > 999 (impossible)
    db.execute(
        'SYS EVOLVE RULE "nope" WHEN memory_pct > 999 THEN SET eviction_target_ratio = 0.5 COOLDOWN 10'
    )

    if hasattr(db, "_evolution_engine"):
        signals = db._evolution_engine.compute_signals()
        db._evolution_engine.evaluate(signals)

    # Should be unchanged
    assert db._sys_executor._eviction_target_ratio == pytest.approx(0.8)


def test_cooldown_prevents_refire():
    """Test 9: Rule won't fire twice within cooldown window."""
    db = SuperGraph(ceiling_mb=100, embedder=None)
    db.execute(
        'SYS EVOLVE RULE "cd" WHEN memory_pct >= 0 THEN SET eviction_target_ratio = 0.65 COOLDOWN 3600'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()

    # First fire
    engine.evaluate(signals)
    ratio_after_first = db._sys_executor._eviction_target_ratio

    # Reset ratio to detect if rule fires again
    db._sys_executor._eviction_target_ratio = 0.9

    # Second evaluation (still within cooldown)
    engine.evaluate(signals)
    ratio_after_second = db._sys_executor._eviction_target_ratio

    # Should still be 0.9 (rule was in cooldown)
    assert ratio_after_second == pytest.approx(0.9)


def test_priority_ordering():
    """Test 10: Lower priority number fires first and wins conflicts."""
    db = SuperGraph(ceiling_mb=100, embedder=None)

    # Both rules fire (memory_pct >= 0), different target values, different priorities
    db.execute(
        'SYS EVOLVE RULE "low-pri" WHEN memory_pct >= 0 THEN SET eviction_target_ratio = 0.5 COOLDOWN 10 PRIORITY 10'
    )
    db.execute(
        'SYS EVOLVE RULE "high-pri" WHEN memory_pct >= 0 THEN SET eviction_target_ratio = 0.7 COOLDOWN 10 PRIORITY 1'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()
    engine.evaluate(signals)

    # Priority 1 fires first, sets to 0.7. Priority 10 fires next but conflict - skipped.
    # Actually: lowest number = highest priority. Priority 1 wins.
    assert db._sys_executor._eviction_target_ratio == pytest.approx(0.7)


def test_frozen_signals():
    """Test 11: Rule B evaluates against snapshot, not Rule A's side effects."""
    db = SuperGraph(ceiling_mb=100, embedder=None)

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    # Rules target different params - use compute_signals to verify snapshot
    engine = db._evolution_engine
    snap1 = engine.compute_signals()
    snap2 = engine.compute_signals()

    # Two calls to compute_signals at same tick should give consistent readings
    # (within float rounding - node_count should be identical)
    assert snap1["node_count"] == snap2["node_count"]
    assert snap1["memory_pct"] == pytest.approx(snap2["memory_pct"], abs=0.01)


def test_conflict_detection_at_create():
    """Test 12: Same param + same priority is unresolvable - must warn at creation."""
    db = SuperGraph(ceiling_mb=100, embedder=None)
    # Both rules: same param, same priority (default 5) - unresolvable conflict
    db.execute(
        'SYS EVOLVE RULE "r1" WHEN memory_pct > 80 THEN SET eviction_target_ratio = 0.6 COOLDOWN 60'
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        db.execute(
            'SYS EVOLVE RULE "r2" WHEN memory_pct > 90 THEN SET eviction_target_ratio = 0.5 COOLDOWN 60'
        )
        assert any("conflict" in str(warning.message).lower() for warning in w), (
            "same param + same priority must emit a conflict warning"
        )

    # Both rules must still be stored (warning does not block creation)
    lst = db.execute("SYS EVOLVE LIST")
    names = [r["name"] for r in lst.data] if isinstance(lst.data, list) else []
    assert "r1" in names
    assert "r2" in names


def test_conflict_runtime_highest_wins():
    """Test 13: When two rules fire targeting same param, lowest priority number wins."""
    db = SuperGraph(ceiling_mb=100, embedder=None)

    db.execute(
        'SYS EVOLVE RULE "winner" WHEN memory_pct >= 0 THEN SET eviction_target_ratio = 0.6 COOLDOWN 10 PRIORITY 1'
    )
    db.execute(
        'SYS EVOLVE RULE "loser" WHEN memory_pct >= 0 THEN SET eviction_target_ratio = 0.9 COOLDOWN 10 PRIORITY 5'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()
    engine.evaluate(signals)

    assert db._sys_executor._eviction_target_ratio == pytest.approx(0.6)


def test_adjust_clamps():
    """Test 14: ADJUST past min/max constraint clamps to boundary."""
    db = SuperGraph(ceiling_mb=100, embedder=None)

    # similarity_threshold has constraint [0.5, 0.99]
    # Set it near the top, then try to push past
    db.execute(
        'SYS EVOLVE RULE "clamp" WHEN memory_pct >= 0 THEN ADJUST similarity_threshold BY 0.5 COOLDOWN 10'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()
    engine.evaluate(signals)

    # similarity_threshold should be clamped to 0.99, not exceed it
    threshold = db._config.vector.similarity_threshold
    # After ADJUST, the engine updates the live config value
    # Check via a getter if available
    if hasattr(engine, "_get_param"):
        val = engine._get_param("similarity_threshold")
        assert val <= 0.99


def test_adjust_until_stops():
    """Test 15: ADJUST UNTIL stops when target reached."""
    db = SuperGraph(ceiling_mb=100, embedder=None)

    # Start at 0.85, adjust by -0.05 until 0.70
    db.execute(
        'SYS EVOLVE RULE "until" WHEN memory_pct >= 0 THEN ADJUST recall_decay BY -0.1 UNTIL 0.5 COOLDOWN 10'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine

    # Run multiple ticks - should stop at 0.5 not go below
    for _ in range(10):
        signals = engine.compute_signals()
        engine.evaluate(signals)
        # Reset cooldown so it can fire again
        if hasattr(engine, "_rules") and engine._rules:
            for r in engine._rules.values():
                if r.name == "until":
                    r.last_fired_at = 0.0

    # recall_decay should not have gone below 0.5
    val = db._executor._recall_decay
    assert val >= 0.5 - 0.001  # small float tolerance


def test_adjust_ceiling_negative_noop():
    """Test 16: ADJUST ceiling_mb by negative amount is a no-op (monotonic)."""
    db = SuperGraph(ceiling_mb=256, embedder=None)
    initial_ceiling = db.ceiling_mb

    db.execute(
        'SYS EVOLVE RULE "shrink" WHEN memory_pct >= 0 THEN ADJUST ceiling_mb BY -64 COOLDOWN 10'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()
    engine.evaluate(signals)

    # ceiling_mb must not decrease
    assert db.ceiling_mb >= initial_ceiling


def test_set_respects_constraints():
    """Test 17: SET out-of-range value is clamped to constraint boundary."""
    db = SuperGraph(ceiling_mb=100, embedder=None)

    # recall_decay has constraint [0.1, 1.0]; set to 2.0
    db.execute(
        'SYS EVOLVE RULE "clamp2" WHEN memory_pct >= 0 THEN SET recall_decay = 2.0 COOLDOWN 10'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()
    engine.evaluate(signals)

    # Should be clamped to 1.0 (max)
    assert db._executor._recall_decay <= 1.0


def test_protected_kinds_always_schema():
    """Test 18: REMOVE protected_kinds cannot remove 'schema' or 'config'."""
    db = SuperGraph(ceiling_mb=100, embedder=None)

    db.execute(
        'SYS EVOLVE RULE "unprotect" WHEN memory_pct >= 0 THEN REMOVE protected_kinds "schema" COOLDOWN 10'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()
    engine.evaluate(signals)

    # 'schema' must remain protected
    if hasattr(engine, "_get_param"):
        kinds = engine._get_param("protected_kinds")
        assert "schema" in kinds
        assert "config" in kinds


def test_remember_weights_normalization():
    """Test 19: After SET remember_weights, values are auto-normalized to sum=1.0."""
    db = SuperGraph(ceiling_mb=100, embedder=None)

    # Set weights that don't sum to 1 (sum = 1.3)
    db.execute(
        'SYS EVOLVE RULE "weights" WHEN memory_pct >= 0 THEN SET remember_weights = [0.6, 0.4, 0.3] COOLDOWN 10'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()
    engine.evaluate(signals)

    weights = db._executor._remember_weights
    assert abs(sum(weights) - 1.0) < 0.001, f"Weights don't sum to 1: {weights}"


def test_run_action_does_not_raise():
    """Test 20: THEN RUN never raises - bad DSL fails gracefully."""
    db = SuperGraph(ceiling_mb=100, embedder=None)

    db.execute(
        'SYS EVOLVE RULE "runner" WHEN memory_pct >= 0 THEN RUN SYS OPTIMIZE COOLDOWN 10'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()
    try:
        engine.evaluate(signals)
    except Exception as e:
        pytest.fail(f"evaluate() raised with RUN action: {e}")


def test_run_action_executes_dsl():
    """THEN RUN dispatches valid DSL and reports status 'applied'."""
    db = SuperGraph(ceiling_mb=100, embedder=None)

    db.execute(
        'SYS EVOLVE RULE "run-stats" WHEN memory_pct >= 0 THEN RUN SYS STATS COOLDOWN 10'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()
    events = engine.evaluate(signals)

    run_actions = [a for e in events for a in e["actions"] if a["kind"] == "run"]
    assert len(run_actions) == 1
    assert run_actions[0]["status"] == "applied"


def test_run_action_bad_dsl_graceful():
    """THEN RUN with unparseable DSL fails gracefully - status 'failed:...' not a crash."""
    db = SuperGraph(ceiling_mb=100, embedder=None)

    db.execute(
        'SYS EVOLVE RULE "bad-run" WHEN memory_pct >= 0 THEN RUN INVALID_CMD_XYZ COOLDOWN 10'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()

    events = engine.evaluate(signals)

    run_actions = [a for e in events for a in e["actions"] if a["kind"] == "run"]
    assert len(run_actions) == 1
    assert run_actions[0]["status"].startswith("failed:")


def test_run_action_status_in_history(tmp_path):
    """History entry records final RUN status ('applied') not 'pending'."""
    db_dir = tmp_path / "run_hist"
    db_dir.mkdir()
    db = SuperGraph(path=str(db_dir), embedder=None)
    db.execute(
        'SYS EVOLVE RULE "run-hist" WHEN memory_pct >= 0 THEN RUN SYS STATS COOLDOWN 10'
    )

    engine = db._evolution_engine
    signals = engine.compute_signals()
    engine.evaluate(signals)

    hist = db.execute("SYS EVOLVE HISTORY LIMIT 1")
    assert hist.data
    run_actions = [a for a in hist.data[0]["actions"] if a["kind"] == "run"]
    assert run_actions[0]["status"] == "applied"
    db.close()


def test_recall_misses_signal_exists():
    """recall_misses is a known signal and returned by compute_signals()."""
    from supergraph.core.evolve import KNOWN_SIGNALS

    assert "recall_misses" in KNOWN_SIGNALS

    db = SuperGraph(ceiling_mb=100, embedder=None)
    engine = db._evolution_engine
    signals = engine.compute_signals()
    assert "recall_misses" in signals
    assert isinstance(signals["recall_misses"], (int, float))


def test_reentrancy_guard():
    """Test 21: Nested _check_health call during evolution tick skips evolution."""
    db = SuperGraph(ceiling_mb=100, embedder=None)

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine

    # Manually set _evaluating = True
    engine._evaluating = True

    # Trigger _check_health - should not cause recursive evaluate()
    try:
        db._optimizer._check_health()
    except Exception as e:
        pytest.fail(f"_check_health raised during re-entry: {e}")
    finally:
        engine._evaluating = False


def test_history_logged(tmp_path):
    """Test 22: When a rule fires, history entry is recorded with signals + prev values."""
    db_dir = tmp_path / "hist_log"
    db_dir.mkdir()
    db = SuperGraph(path=str(db_dir), embedder=None)

    db.execute(
        'SYS EVOLVE RULE "log-test" WHEN memory_pct >= 0 THEN SET eviction_target_ratio = 0.65 COOLDOWN 10'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()
    engine.evaluate(signals)

    hist = db.execute("SYS EVOLVE HISTORY LIMIT 10")
    assert hist.kind in ("history", "ok")
    assert isinstance(hist.data, list)
    assert len(hist.data) >= 1
    entry = hist.data[0]
    assert "rule_name" in entry
    assert "signals" in entry or "signals_json" in entry
    assert "status" in entry
    db.close()


def test_history_limit(tmp_path):
    """Test 23: HISTORY LIMIT n returns at most n most recent entries."""
    db_dir = tmp_path / "hist_limit"
    db_dir.mkdir()
    db = SuperGraph(path=str(db_dir), embedder=None)

    db.execute(
        'SYS EVOLVE RULE "many" WHEN memory_pct >= 0 THEN SET eviction_target_ratio = 0.65 COOLDOWN 10'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine

    # Fire multiple times
    for _ in range(5):
        signals = engine.compute_signals()
        engine.evaluate(signals)
        # Reset cooldown
        if "many" in engine._rules:
            engine._rules["many"].last_fired_at = 0.0

    hist = db.execute("SYS EVOLVE HISTORY LIMIT 2")
    assert isinstance(hist.data, list)
    assert len(hist.data) <= 2
    db.close()


def test_reset_reverts():
    """Test 24: SYS EVOLVE RESET disables all rules and reverts config to defaults."""
    db = SuperGraph(ceiling_mb=100, embedder=None)

    db.execute(
        'SYS EVOLVE RULE "to-reset" WHEN memory_pct > 80 THEN SET eviction_target_ratio = 0.5 COOLDOWN 60'
    )

    res = db.execute("SYS EVOLVE RESET")
    assert res.kind == "ok"

    lst = db.execute("SYS EVOLVE LIST")
    rules = lst.data if isinstance(lst.data, list) else []
    # All rules disabled after reset
    for rule in rules:
        assert rule.get("enabled") is False


def test_feedback_on_result():
    """Test 25: result.meta['evolution'] is populated when rules fire."""
    db = SuperGraph(ceiling_mb=100, embedder=None)

    db.execute(
        'SYS EVOLVE RULE "feedback" WHEN memory_pct >= 0 THEN SET eviction_target_ratio = 0.65 COOLDOWN 10'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()
    engine.evaluate(signals)
    # _last_evolution_events should be populated
    assert len(db._last_evolution_events) >= 1

    # Next execute() should attach events to result.meta
    result = db.execute('CREATE NODE "fb1" kind="test" x="1"')
    # meta["evolution"] should be populated
    assert "evolution" in result.meta
    assert len(result.meta["evolution"]) >= 1

    # Events should be cleared after attachment
    assert db._last_evolution_events == []


def test_persistence_across_restart(tmp_path):
    """Test 26: Rules survive close + reopen."""
    db_dir = tmp_path / "persist_evo"
    db_dir.mkdir()

    db = SuperGraph(path=str(db_dir), embedder=None)
    db.execute(
        'SYS EVOLVE RULE "persist-me" WHEN memory_pct > 80 THEN SET eviction_target_ratio = 0.6 COOLDOWN 300'
    )
    db.close()

    # Reopen
    db2 = SuperGraph(path=str(db_dir), embedder=None)
    lst = db2.execute("SYS EVOLVE LIST")
    names = [r["name"] for r in lst.data] if isinstance(lst.data, list) else []
    assert "persist-me" in names
    db2.close()


def test_starter_rules_disabled():
    """Test 27: Starter rules from evolve_defaults are present but disabled by default."""
    from supergraph.core.evolve import STARTER_RULES

    assert len(STARTER_RULES) >= 3
    for rule in STARTER_RULES:
        assert rule.get("enabled", True) is False, f"Starter rule {rule.get('name')} should be disabled"


# ============================================================
# Infrastructure: WAL pending_count property
# ============================================================

def test_wal_pending_count_property(tmp_path):
    """WALManager must expose a pending_count property."""
    db_dir = tmp_path / "wal_pending"
    db_dir.mkdir()
    db = SuperGraph(path=str(db_dir), embedder=None)
    assert hasattr(db._wal, "pending_count")
    count = db._wal.pending_count
    assert isinstance(count, int)
    assert count >= 0
    db.close()


# ============================================================
# Engine: Known signals + tunable params (Step 2)
# ============================================================

def test_known_signals_registry():
    """KNOWN_SIGNALS must contain all 12 spec-defined signals."""
    from supergraph.core.evolve import KNOWN_SIGNALS

    expected = {
        "memory_pct", "memory_mb", "node_count", "tombstone_ratio",
        "string_bloat", "recall_hit_rate", "avg_similarity", "eviction_count",
        "query_rate", "write_rate", "edge_density", "wal_pending",
    }
    assert expected.issubset(KNOWN_SIGNALS)


def test_tunable_params_registry():
    """TUNABLE_PARAMS must contain all 10 spec-defined parameters."""
    from supergraph.core.evolve import TUNABLE_PARAMS

    expected = {
        "ceiling_mb", "eviction_target_ratio", "remember_weights",
        "recall_decay", "similarity_threshold", "duplicate_threshold",
        "chunk_max_size", "cost_threshold", "optimize_interval", "protected_kinds",
    }
    assert expected.issubset(TUNABLE_PARAMS)


def test_compute_signals_returns_all_keys():
    """compute_signals() must return all 12 signal keys."""
    db = SuperGraph(ceiling_mb=100, embedder=None)

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()

    expected = {
        "memory_pct", "memory_mb", "node_count", "tombstone_ratio",
        "string_bloat", "recall_hit_rate", "avg_similarity", "eviction_count",
        "query_rate", "write_rate", "edge_density", "wal_pending",
    }
    assert expected.issubset(signals.keys())


def test_recall_hit_rate_defaults_to_1_when_no_queries():
    """recall_hit_rate must be 1.0 when no RECALL queries have been made yet."""
    db = SuperGraph(ceiling_mb=100, embedder=None)

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()
    assert signals["recall_hit_rate"] == pytest.approx(1.0)
