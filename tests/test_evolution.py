import time
import warnings
import pytest

from supergraph import SuperGraph
from supergraph.core.types import Result

pytestmark = pytest.mark.slow


def test_result_meta_field_exists():
    r = Result(kind="ok", data=None, count=0)
    assert hasattr(r, "meta")
    assert r.meta == {}


def test_result_meta_in_to_dict_when_non_empty():
    r = Result(kind="ok", data=None, count=0, meta={"evolution": [{"rule": "x"}]})
    d = r.to_dict()
    assert "meta" in d
    assert d["meta"] == {"evolution": [{"rule": "x"}]}


def test_result_meta_absent_from_to_dict_when_empty():
    r = Result(kind="ok", data=None, count=0)
    d = r.to_dict()
    assert "meta" not in d


def test_evolution_config_exists_in_supergraph_config():
    from supergraph.config import SuperGraphConfig
    cfg = SuperGraphConfig()
    assert hasattr(cfg, "evolution")


def test_evolution_config_defaults():
    from supergraph.config import EvolutionConfig
    cfg = EvolutionConfig()
    assert cfg.similarity_buffer_size == 100
    assert cfg.max_rules == 50
    assert cfg.min_cooldown == 10
    assert cfg.history_retention == 1000


def test_supergraph_has_counters():
    db = SuperGraph(ceiling_mb=100, embedder=None)
    assert hasattr(db, "_counters")
    assert isinstance(db._counters, dict)


def test_counters_track_execute_ok():
    db = SuperGraph(ceiling_mb=100, embedder=None)
    initial = db._counters.get("execute_ok", 0)
    db.execute('CREATE NODE "c1" kind="test" x="1"')
    assert db._counters.get("execute_ok", 0) == initial + 1


def test_counters_track_execute_err():
    db = SuperGraph(ceiling_mb=100, embedder=None)
    initial = db._counters.get("execute_err", 0)
    try:
        db.execute("TOTALLY INVALID QUERY $$$$")
    except Exception:
        pass
    assert db._counters.get("execute_err", 0) == initial + 1


def test_supergraph_has_start_time():
    db = SuperGraph(ceiling_mb=100, embedder=None)
    assert hasattr(db, "_start_time")
    assert isinstance(db._start_time, float)
    assert db._start_time <= time.time()


def test_supergraph_has_similarity_buffer():
    from collections import deque
    db = SuperGraph(ceiling_mb=100, embedder=None)
    assert hasattr(db, "_similarity_buffer")
    assert isinstance(db._similarity_buffer, deque)


def test_supergraph_has_last_evolution_events():
    db = SuperGraph(ceiling_mb=100, embedder=None)
    assert hasattr(db, "_last_evolution_events")
    assert isinstance(db._last_evolution_events, list)


def test_db_evolution_tables_created(tmp_path):
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
    db_dir = tmp_path / "evo_hist_schema"
    db_dir.mkdir()
    db = SuperGraph(path=str(db_dir), embedder=None)
    conn = db._conn

    cols = {row[1] for row in conn.execute("PRAGMA table_info(evolution_history)").fetchall()}
    for expected in ("id", "timestamp", "rule_name", "signals_json", "actions_json", "prev_values_json", "status"):
        assert expected in cols, f"Missing column: {expected}"
    db.close()


def test_create_rule():
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
    db = SuperGraph(ceiling_mb=100, embedder=None)
    res = db.execute(
        'SYS EVOLVE RULE "bad" WHEN bogus_signal > 5 THEN SET eviction_target_ratio = 0.7 COOLDOWN 60'
    )
    assert res.kind == "error"
    assert "unknown signal" in res.data.lower() or "unknown" in res.data.lower()


def test_create_invalid_param():
    db = SuperGraph(ceiling_mb=100, embedder=None)
    res = db.execute(
        'SYS EVOLVE RULE "bad2" WHEN memory_pct > 50 THEN SET fake_param = 0.5 COOLDOWN 60'
    )
    assert res.kind == "error"
    assert "unknown parameter" in res.data.lower() or "unknown" in res.data.lower()


def test_enable_disable_cycle():
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
    db = SuperGraph(ceiling_mb=100, embedder=None)
    initial_ratio = db._sys_executor._eviction_target_ratio

    db.execute(
        'SYS EVOLVE RULE "fire" WHEN memory_pct >= 0 THEN SET eviction_target_ratio = 0.65 COOLDOWN 10'
    )

    if hasattr(db, "_evolution_engine"):
        signals = db._evolution_engine.compute_signals()
        db._evolution_engine.evaluate(signals)

    new_ratio = db._sys_executor._eviction_target_ratio
    assert new_ratio == pytest.approx(0.65)


def test_rule_skips_false_condition():
    db = SuperGraph(ceiling_mb=100, embedder=None)

    db._sys_executor._eviction_target_ratio = 0.8

    db.execute(
        'SYS EVOLVE RULE "nope" WHEN memory_pct > 999 THEN SET eviction_target_ratio = 0.5 COOLDOWN 10'
    )

    if hasattr(db, "_evolution_engine"):
        signals = db._evolution_engine.compute_signals()
        db._evolution_engine.evaluate(signals)

    assert db._sys_executor._eviction_target_ratio == pytest.approx(0.8)


def test_cooldown_prevents_refire():
    db = SuperGraph(ceiling_mb=100, embedder=None)
    db.execute(
        'SYS EVOLVE RULE "cd" WHEN memory_pct >= 0 THEN SET eviction_target_ratio = 0.65 COOLDOWN 3600'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()

    engine.evaluate(signals)
    ratio_after_first = db._sys_executor._eviction_target_ratio

    db._sys_executor._eviction_target_ratio = 0.9

    engine.evaluate(signals)
    ratio_after_second = db._sys_executor._eviction_target_ratio

    assert ratio_after_second == pytest.approx(0.9)


def test_priority_ordering():
    db = SuperGraph(ceiling_mb=100, embedder=None)

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

    assert db._sys_executor._eviction_target_ratio == pytest.approx(0.7)


def test_frozen_signals():
    db = SuperGraph(ceiling_mb=100, embedder=None)

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    snap1 = engine.compute_signals()
    snap2 = engine.compute_signals()

    assert snap1["node_count"] == snap2["node_count"]
    assert snap1["memory_pct"] == pytest.approx(snap2["memory_pct"], abs=0.01)


def test_conflict_detection_at_create():
    db = SuperGraph(ceiling_mb=100, embedder=None)
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

    lst = db.execute("SYS EVOLVE LIST")
    names = [r["name"] for r in lst.data] if isinstance(lst.data, list) else []
    assert "r1" in names
    assert "r2" in names


def test_conflict_runtime_highest_wins():
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
    db = SuperGraph(ceiling_mb=100, embedder=None)

    db.execute(
        'SYS EVOLVE RULE "clamp" WHEN memory_pct >= 0 THEN ADJUST similarity_threshold BY 0.5 COOLDOWN 10'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()
    engine.evaluate(signals)

    threshold = db._config.vector.similarity_threshold
    if hasattr(engine, "_get_param"):
        val = engine._get_param("similarity_threshold")
        assert val <= 0.99


def test_adjust_until_stops():
    db = SuperGraph(ceiling_mb=100, embedder=None)

    db.execute(
        'SYS EVOLVE RULE "until" WHEN memory_pct >= 0 THEN ADJUST recall_decay BY -0.1 UNTIL 0.5 COOLDOWN 10'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine

    for _ in range(10):
        signals = engine.compute_signals()
        engine.evaluate(signals)
        if hasattr(engine, "_rules") and engine._rules:
            for r in engine._rules.values():
                if r.name == "until":
                    r.last_fired_at = 0.0

    val = db._executor._recall_decay
    assert val >= 0.5 - 0.001


def test_adjust_ceiling_negative_noop():
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

    assert db.ceiling_mb >= initial_ceiling


def test_set_respects_constraints():
    db = SuperGraph(ceiling_mb=100, embedder=None)

    db.execute(
        'SYS EVOLVE RULE "clamp2" WHEN memory_pct >= 0 THEN SET recall_decay = 2.0 COOLDOWN 10'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()
    engine.evaluate(signals)

    assert db._executor._recall_decay <= 1.0


def test_protected_kinds_always_schema():
    db = SuperGraph(ceiling_mb=100, embedder=None)

    db.execute(
        'SYS EVOLVE RULE "unprotect" WHEN memory_pct >= 0 THEN REMOVE protected_kinds "schema" COOLDOWN 10'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()
    engine.evaluate(signals)

    if hasattr(engine, "_get_param"):
        kinds = engine._get_param("protected_kinds")
        assert "schema" in kinds
        assert "config" in kinds


def test_remember_weights_normalization():
    db = SuperGraph(ceiling_mb=100, embedder=None)

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
    from supergraph.core.evolve import KNOWN_SIGNALS

    assert "recall_misses" in KNOWN_SIGNALS

    db = SuperGraph(ceiling_mb=100, embedder=None)
    engine = db._evolution_engine
    signals = engine.compute_signals()
    assert "recall_misses" in signals
    assert isinstance(signals["recall_misses"], (int, float))


def test_reentrancy_guard():
    db = SuperGraph(ceiling_mb=100, embedder=None)

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine

    engine._evaluating = True

    try:
        db._optimizer._check_health()
    except Exception as e:
        pytest.fail(f"_check_health raised during re-entry: {e}")
    finally:
        engine._evaluating = False


def test_history_logged(tmp_path):
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
    db_dir = tmp_path / "hist_limit"
    db_dir.mkdir()
    db = SuperGraph(path=str(db_dir), embedder=None)

    db.execute(
        'SYS EVOLVE RULE "many" WHEN memory_pct >= 0 THEN SET eviction_target_ratio = 0.65 COOLDOWN 10'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine

    for _ in range(5):
        signals = engine.compute_signals()
        engine.evaluate(signals)
        if "many" in engine._rules:
            engine._rules["many"].last_fired_at = 0.0

    hist = db.execute("SYS EVOLVE HISTORY LIMIT 2")
    assert isinstance(hist.data, list)
    assert len(hist.data) <= 2
    db.close()


def test_reset_reverts():
    db = SuperGraph(ceiling_mb=100, embedder=None)

    db.execute(
        'SYS EVOLVE RULE "to-reset" WHEN memory_pct > 80 THEN SET eviction_target_ratio = 0.5 COOLDOWN 60'
    )

    res = db.execute("SYS EVOLVE RESET")
    assert res.kind == "ok"

    lst = db.execute("SYS EVOLVE LIST")
    rules = lst.data if isinstance(lst.data, list) else []
    for rule in rules:
        assert rule.get("enabled") is False


def test_feedback_on_result():
    db = SuperGraph(ceiling_mb=100, embedder=None)

    db.execute(
        'SYS EVOLVE RULE "feedback" WHEN memory_pct >= 0 THEN SET eviction_target_ratio = 0.65 COOLDOWN 10'
    )

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()
    engine.evaluate(signals)
    assert len(db._last_evolution_events) >= 1

    result = db.execute('CREATE NODE "fb1" kind="test" x="1"')
    assert "evolution" in result.meta
    assert len(result.meta["evolution"]) >= 1

    assert db._last_evolution_events == []


def test_persistence_across_restart(tmp_path):
    db_dir = tmp_path / "persist_evo"
    db_dir.mkdir()

    db = SuperGraph(path=str(db_dir), embedder=None)
    db.execute(
        'SYS EVOLVE RULE "persist-me" WHEN memory_pct > 80 THEN SET eviction_target_ratio = 0.6 COOLDOWN 300'
    )
    db.close()

    db2 = SuperGraph(path=str(db_dir), embedder=None)
    lst = db2.execute("SYS EVOLVE LIST")
    names = [r["name"] for r in lst.data] if isinstance(lst.data, list) else []
    assert "persist-me" in names
    db2.close()


def test_starter_rules_disabled():
    from supergraph.core.evolve import STARTER_RULES

    assert len(STARTER_RULES) >= 3
    for rule in STARTER_RULES:
        assert rule.get("enabled", True) is False, f"Starter rule {rule.get('name')} should be disabled"


def test_wal_pending_count_property(tmp_path):
    db_dir = tmp_path / "wal_pending"
    db_dir.mkdir()
    db = SuperGraph(path=str(db_dir), embedder=None)
    assert hasattr(db._wal, "pending_count")
    count = db._wal.pending_count
    assert isinstance(count, int)
    assert count >= 0
    db.close()


def test_known_signals_registry():
    from supergraph.core.evolve import KNOWN_SIGNALS

    expected = {
        "memory_pct", "memory_mb", "node_count", "tombstone_ratio",
        "string_bloat", "recall_hit_rate", "avg_similarity", "eviction_count",
        "query_rate", "write_rate", "edge_density", "wal_pending",
    }
    assert expected.issubset(KNOWN_SIGNALS)


def test_tunable_params_registry():
    from supergraph.core.evolve import TUNABLE_PARAMS

    expected = {
        "ceiling_mb", "eviction_target_ratio", "remember_weights",
        "recall_decay", "similarity_threshold", "duplicate_threshold",
        "chunk_max_size", "cost_threshold", "optimize_interval", "protected_kinds",
    }
    assert expected.issubset(TUNABLE_PARAMS)


def test_compute_signals_returns_all_keys():
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
    db = SuperGraph(ceiling_mb=100, embedder=None)

    if not hasattr(db, "_evolution_engine"):
        pytest.skip("EvolutionEngine not wired yet")

    engine = db._evolution_engine
    signals = engine.compute_signals()
    assert signals["recall_hit_rate"] == pytest.approx(1.0)
