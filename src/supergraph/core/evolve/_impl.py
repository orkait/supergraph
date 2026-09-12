"""Metacognitive evolution engine: agent-writable WHEN/THEN rules that tune
the engine's own memory behavior during health-check cycles.
"""

import json
import time
import logging
import warnings
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation registries
# ---------------------------------------------------------------------------

KNOWN_SIGNALS: frozenset = frozenset({
    "memory_pct",
    "memory_mb",
    "node_count",
    "tombstone_ratio",
    "string_bloat",
    "recall_hit_rate",
    "recall_misses",
    "avg_similarity",
    "eviction_count",
    "query_rate",
    "write_rate",
    "edge_density",
    "wal_pending",
})

# name → constraint dict: type, min, max, and optional special flags
TUNABLE_PARAMS: dict = {
    "ceiling_mb":            {"type": int,   "min": 32,    "max": None,       "monotonic": True},
    "eviction_target_ratio": {"type": float, "min": 0.5,   "max": 0.95},
    "remember_weights":      {"type": list,  "min": None,  "max": None,       "normalize": True, "length_range": [3, 4]},
    "recall_decay":          {"type": float, "min": 0.1,   "max": 1.0},
    "similarity_threshold":  {"type": float, "min": 0.5,   "max": 0.99},
    "duplicate_threshold":   {"type": float, "min": 0.8,   "max": 1.0},
    "chunk_max_size":        {"type": int,   "min": 200,   "max": 10_000},
    "cost_threshold":        {"type": int,   "min": 1_000, "max": 10_000_000},
    "optimize_interval":     {"type": int,   "min": 50,    "max": 10_000},
    "protected_kinds":       {"type": set,   "min": None,  "max": None,       "always_includes": {"schema", "config"}},
}

# Protected kinds that can never be removed
_ALWAYS_PROTECTED = frozenset({"schema", "config"})

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    signal: str
    operator: str   # >, <, >=, <=, ==, !=
    value: float


@dataclass
class Action:
    kind: str           # set | adjust | add | remove | run
    param: str
    value: object = None
    delta: float = 0.0
    until: object = None    # ADJUST UNTIL target


@dataclass
class EvolutionRule:
    name: str
    conditions: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    cooldown: int = 60
    priority: int = 5
    enabled: bool = True
    last_fired_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "conditions": [{"signal": c.signal, "operator": c.operator, "value": c.value}
                           for c in self.conditions],
            "actions": [{"kind": a.kind, "param": a.param, "value": a.value,
                         "delta": a.delta, "until": a.until}
                        for a in self.actions],
            "cooldown": self.cooldown,
            "priority": self.priority,
            "enabled": self.enabled,
            "last_fired_at": self.last_fired_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvolutionRule":
        r = cls(
            name=d["name"],
            cooldown=d.get("cooldown", 60),
            priority=d.get("priority", 5),
            enabled=d.get("enabled", True),
            last_fired_at=d.get("last_fired_at", 0.0),
        )
        r.conditions = [Condition(**c) for c in d.get("conditions", [])]
        r.actions = [Action(**a) for a in d.get("actions", [])]
        return r


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_condition(cond: Condition, signals: dict) -> bool:
    val = signals.get(cond.signal, 0)
    op = cond.operator
    if op == ">":  return val > cond.value
    if op == "<":  return val < cond.value
    if op == ">=": return val >= cond.value
    if op == "<=": return val <= cond.value
    if op == "==": return val == cond.value
    if op == "!=": return val != cond.value
    return False


def _clamp(value, lo, hi):
    if lo is not None and value < lo:
        return lo
    if hi is not None and value > hi:
        return hi
    return value


def _normalize_weights(weights: list) -> list:
    total = sum(weights)
    if total == 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


# ---------------------------------------------------------------------------
# EvolutionEngine
# ---------------------------------------------------------------------------

class EvolutionEngine:
    """Evaluates WHEN/THEN rules during health-check ticks and tunes config."""

    def __init__(self, gs, conn, config):
        """
        Args:
            gs: SuperGraph instance (live ref; not weakref - engine lifetime == gs lifetime)
            conn: sqlite3.Connection or None (in-memory mode)
            config: EvolutionConfig
        """
        self._gs = gs
        self._conn = conn
        self._config = config
        self._rules: dict[str, EvolutionRule] = {}
        self._evaluating = False

        # Params with no live runtime field - tracked here
        self._live_params: dict = {}

        self._load_from_db()

    # -----------------------------------------------------------------------
    # Rule management
    # -----------------------------------------------------------------------

    def add_rule(self, rule: EvolutionRule) -> None | str:
        """Validate and store rule. Returns error string or None on success."""
        # Validate signal names
        for cond in rule.conditions:
            if cond.signal not in KNOWN_SIGNALS:
                return f"unknown signal: '{cond.signal}'"
            if cond.operator in ("==", "!="):
                warnings.warn(
                    f"Rule '{rule.name}': float equality on signal '{cond.signal}' is discouraged",
                    UserWarning, stacklevel=3,
                )

        # Validate param names
        for action in rule.actions:
            if action.kind not in ("run",) and action.param not in TUNABLE_PARAMS:
                return f"unknown parameter: '{action.param}'"

        # Check duplicate name
        if rule.name in self._rules:
            return f"rule already exists: '{rule.name}' (duplicate name)"

        # Check max rules
        if len(self._rules) >= self._config.max_rules:
            return f"max rules limit ({self._config.max_rules}) reached"

        # Cooldown floor
        if rule.cooldown < self._config.min_cooldown:
            rule.cooldown = self._config.min_cooldown

        # Conflict detection: warn only when same param + same priority - that is
        # a genuinely unresolvable conflict (execution order becomes insertion-
        # dependent). Different priorities on the same param is intentional design;
        # the priority system resolves it at runtime.
        for action in rule.actions:
            for er in self._rules.values():
                if not er.enabled or er.priority != rule.priority:
                    continue
                for ea in er.actions:
                    if ea.kind == action.kind and ea.param == action.param:
                        warnings.warn(
                            f"Rule '{rule.name}': unresolvable conflict with rule '{er.name}' -"
                            f" both target param '{action.param}' via '{action.kind}' at priority {rule.priority}",
                            UserWarning, stacklevel=3,
                        )
                        break

        self._rules[rule.name] = rule
        self._persist_rule(rule)
        return None

    def enable_rule(self, name: str) -> bool:
        if name not in self._rules:
            return False
        self._rules[name].enabled = True
        self._persist_rule(self._rules[name])
        return True

    def disable_rule(self, name: str) -> bool:
        if name not in self._rules:
            return False
        self._rules[name].enabled = False
        self._persist_rule(self._rules[name])
        return True

    def delete_rule(self, name: str) -> bool:
        if name not in self._rules:
            return False
        del self._rules[name]
        if self._conn is not None:
            self._conn.execute("DELETE FROM evolution_rules WHERE name=?", (name,))
            self._conn.commit()
        return True

    def list_rules(self) -> list[dict]:
        return [r.to_dict() for r in sorted(self._rules.values(), key=lambda r: r.priority)]

    def get_rule(self, name: str) -> dict | None:
        r = self._rules.get(name)
        return r.to_dict() if r else None

    def reset(self) -> None:
        """Disable all rules. (Config revert is left to the caller.)"""
        for rule in self._rules.values():
            rule.enabled = False
            self._persist_rule(rule)

    # -----------------------------------------------------------------------
    # Signal computation
    # -----------------------------------------------------------------------

    def compute_signals(self) -> dict:
        gs = self._gs
        store = gs._store

        # Memory
        try:
            from supergraph.core.memory import measure
            m = measure(store, gs._vector_store, gs._document_store)
            total_bytes = m.get("total", 0)
        except Exception as err:
            logger.debug("memory.measure() failed during evolve telemetry: %s", err)
            total_bytes = 0

        ceiling = getattr(store, "_ceiling_bytes", 1) or 1
        memory_pct = total_bytes / ceiling * 100
        memory_mb = total_bytes / 1_000_000

        # Node count
        node_count = store.node_count

        # Health (tombstone_ratio, string_bloat)
        try:
            from supergraph.core.optimizer import health_check
            h = health_check(store, gs._vector_store, gs._document_store)
            tombstone_ratio = h.get("tombstone_ratio", 0.0)
            string_bloat = h.get("string_bloat", 0.0)
        except Exception as err:
            logger.debug("optimizer.health_check() failed during evolve telemetry: %s", err)
            tombstone_ratio = 0.0
            string_bloat = 0.0

        # Recall hit rate + raw miss counter
        hits = gs._counters.get("recall_hits", 0)
        misses = gs._counters.get("recall_misses", 0)
        recall_hit_rate = hits / max(hits + misses, 1) if (hits + misses) > 0 else 1.0
        recall_misses = misses

        # Avg similarity
        buf = gs._similarity_buffer
        avg_similarity = sum(buf) / max(len(buf), 1) if buf else 0.0

        # Eviction count
        eviction_count = gs._counters.get("eviction_total", 0)

        # Query/write rates
        uptime_minutes = max((time.time() - gs._start_time) / 60.0, 1.0 / 60)
        query_rate = gs._counters.get("execute_ok", 0) / uptime_minutes
        write_counter = getattr(gs._optimizer, "_write_counter", 0)
        write_rate = write_counter / uptime_minutes

        try:
            edge_mats = getattr(store, "edge_matrices", None)
            total_edges = edge_mats.total_edges if edge_mats is not None else 0
        except Exception:
            total_edges = 0
        edge_density = total_edges / max(node_count, 1)

        # WAL pending
        wal_pending = gs._wal.pending_count if gs._conn else 0

        return {
            "memory_pct": memory_pct,
            "memory_mb": memory_mb,
            "node_count": node_count,
            "tombstone_ratio": tombstone_ratio,
            "string_bloat": string_bloat,
            "recall_hit_rate": recall_hit_rate,
            "recall_misses": recall_misses,
            "avg_similarity": avg_similarity,
            "eviction_count": eviction_count,
            "query_rate": query_rate,
            "write_rate": write_rate,
            "edge_density": edge_density,
            "wal_pending": wal_pending,
        }

    # -----------------------------------------------------------------------
    # Param access
    # -----------------------------------------------------------------------

    def _get_param(self, name: str):
        gs = self._gs
        if name == "eviction_target_ratio":
            return gs._sys_executor._eviction_target_ratio
        if name == "remember_weights":
            return list(gs._executor._remember_weights)
        if name == "recall_decay":
            return gs._executor._recall_decay
        if name == "chunk_max_size":
            return gs._executor._chunk_max_size
        if name == "cost_threshold":
            return gs._executor.cost_threshold
        if name == "optimize_interval":
            return gs._optimizer._optimize_interval
        if name == "ceiling_mb":
            return gs._ceiling_bytes // 1_000_000
        if name == "similarity_threshold":
            val = gs._executor._similarity_threshold
            return val if val is not None else gs._config.vector.similarity_threshold
        if name == "duplicate_threshold":
            val = gs._sys_executor._duplicate_threshold_override
            return val if val is not None else gs._config.vector.duplicate_threshold
        if name == "protected_kinds":
            val = gs._sys_executor._protected_kinds
            return val if val is not None else set(gs._config.core.protected_kinds)
        if name in self._live_params:
            return self._live_params[name]
        return None

    def _set_param(self, name: str, value) -> str:
        """Apply value to the live runtime. Returns status string."""
        gs = self._gs
        spec = TUNABLE_PARAMS.get(name, {})

        if name == "ceiling_mb":
            current = gs._ceiling_bytes // 1_000_000
            if value < current:
                return "skipped:monotonic"
            new_bytes = int(value) * 1_000_000
            new_bytes = max(new_bytes, 32 * 1_000_000)
            gs._ceiling_bytes = new_bytes
            gs._store._ceiling_bytes = new_bytes
            return "applied"

        if name == "eviction_target_ratio":
            value = _clamp(value, spec.get("min"), spec.get("max"))
            gs._sys_executor._eviction_target_ratio = float(value)
            return "applied"

        if name == "remember_weights":
            if not isinstance(value, (list, tuple)):
                return "skipped:invalid_length"
            if len(value) not in (3, 4):
                return "skipped:invalid_length"
            value = _normalize_weights(list(float(w) for w in value))
            gs._executor._remember_weights = value
            return "applied"

        if name == "recall_decay":
            value = _clamp(float(value), spec.get("min"), spec.get("max"))
            gs._executor._recall_decay = value
            return "applied"

        if name == "chunk_max_size":
            value = _clamp(int(value), spec.get("min"), spec.get("max"))
            gs._executor._chunk_max_size = value
            return "applied"

        if name == "cost_threshold":
            value = _clamp(int(value), spec.get("min"), spec.get("max"))
            gs._executor.cost_threshold = value
            return "applied"

        if name == "optimize_interval":
            value = _clamp(int(value), spec.get("min"), spec.get("max"))
            gs._optimizer._optimize_interval = value
            return "applied"

        if name == "similarity_threshold":
            value = _clamp(float(value), spec.get("min"), spec.get("max"))
            gs._executor._similarity_threshold = value
            return "applied"

        if name == "duplicate_threshold":
            value = _clamp(float(value), spec.get("min"), spec.get("max"))
            gs._sys_executor._duplicate_threshold_override = value
            return "applied"

        if name == "protected_kinds":
            kinds = set(value) | _ALWAYS_PROTECTED
            gs._sys_executor._protected_kinds = kinds
            return "applied"

        return "skipped:unknown"

    def _adjust_param(self, name: str, delta: float, until=None) -> str:
        """Adjust param by delta, clamped to constraints. Returns status."""
        current = self._get_param(name)
        if current is None:
            return "skipped:unknown"

        spec = TUNABLE_PARAMS.get(name, {})

        # Monotonic: no negative adjustments to ceiling_mb
        if name == "ceiling_mb" and delta < 0:
            return "skipped:monotonic"

        if isinstance(current, (int, float)):
            new_val = current + delta
            # UNTIL: stop if target reached
            if until is not None:
                if delta > 0 and current >= until:
                    return "skipped:until_reached"
                if delta < 0 and current <= until:
                    return "skipped:until_reached"
                new_val = _clamp(new_val, None, until) if delta > 0 else _clamp(new_val, until, None)
            clamped = _clamp(new_val, spec.get("min"), spec.get("max"))
            if clamped != new_val:
                logger.debug("Evolution: ADJUST %s clamped %s → %s", name, new_val, clamped)
            return self._set_param(name, clamped)

        return "skipped:unsupported_type"

    def _add_to_param(self, name: str, element: str) -> str:
        kinds = self._get_param(name)
        if kinds is None:
            kinds = set()
        kinds = set(kinds) | {element}
        if name == "protected_kinds":
            kinds |= _ALWAYS_PROTECTED
            self._gs._sys_executor._protected_kinds = kinds
        else:
            self._live_params[name] = kinds
        return "applied"

    def _remove_from_param(self, name: str, element: str) -> str:
        if name == "protected_kinds":
            if element in _ALWAYS_PROTECTED:
                logger.debug("Evolution: cannot remove always-protected kind '%s'", element)
                return "skipped:protected"
        kinds = self._get_param(name)
        if kinds is None:
            return "skipped:not_found"
        kinds = set(kinds) - {element}
        if name == "protected_kinds":
            kinds |= _ALWAYS_PROTECTED
            self._gs._sys_executor._protected_kinds = kinds
        else:
            self._live_params[name] = kinds
        return "applied"

    # -----------------------------------------------------------------------
    # Core evaluation
    # -----------------------------------------------------------------------

    def evaluate(self, signals: dict) -> list[dict]:
        """Evaluate all enabled rules against frozen signals snapshot.

        Returns list of history events that were applied.
        """
        if self._evaluating:
            return []

        self._evaluating = True
        events = []

        try:
            enabled = sorted(
                (r for r in self._rules.values() if r.enabled),
                key=lambda r: r.priority,
            )

            # Phase 1: collect firing rules (frozen snapshot - no side effects yet)
            pending: list[tuple[EvolutionRule, list[Action]]] = []
            for rule in enabled:
                now = time.time()
                if now - rule.last_fired_at < rule.cooldown:
                    continue
                if all(_check_condition(c, signals) for c in rule.conditions):
                    pending.append((rule, list(rule.actions)))

                # Phase 2: apply actions (conflict = first wins per param+kind)
            claimed: set[tuple[str, str]] = set()
            run_queue: list[tuple[str, str]] = []  # (rule_name, cmd)

            for rule, actions in pending:
                now = time.time()
                prev_values = {}
                applied_actions = []
                rule_events = []

                for action in actions:
                    key = (action.kind, action.param) if action.kind != "run" else ("run", action.param)
                    prev = self._get_param(action.param) if action.kind != "run" else None

                    # Conflict check (only for set/adjust - not run/add/remove)
                    if action.kind in ("set", "adjust") and key in claimed:
                        rule_events.append({"param": action.param, "status": "skipped:conflict"})
                        continue

                    if action.kind == "set":
                        status = self._set_param(action.param, action.value)
                    elif action.kind == "adjust":
                        status = self._adjust_param(action.param, action.delta, action.until)
                    elif action.kind == "add":
                        status = self._add_to_param(action.param, action.value)
                    elif action.kind == "remove":
                        status = self._remove_from_param(action.param, action.value)
                    elif action.kind == "run":
                        run_queue.append((rule.name, action.param))
                        status = "pending"
                    else:
                        status = "skipped:unknown_action"

                    if action.kind not in ("run",):
                        prev_values[action.param] = prev
                    if status == "applied":
                        claimed.add(key)
                    applied_actions.append({"param": action.param, "kind": action.kind, "status": status})
                    rule_events.append({"param": action.param, "status": status})

                rule.last_fired_at = now
                event = {
                    "rule_name": rule.name,
                    "signals": dict(signals),
                    "actions": applied_actions,
                    "prev_values": prev_values,
                    "status": "pending",  # finalized after Phase 3
                }
                events.append(event)

            # Phase 3: execute RUN commands
            # Reentrancy guard (self._evaluating = True) is still active - prevents
            # recursive evolution firing if a RUN action triggers _check_health.
            for rule_name, cmd in run_queue:
                try:
                    self._gs._execute_internal(cmd)
                    run_status = "applied"
                    logger.debug("Evolution RUN '%s' executed (rule '%s')", cmd, rule_name)
                except Exception as e:
                    run_status = f"failed:{type(e).__name__}"
                    logger.warning("Evolution RUN '%s' failed (rule '%s'): %s", cmd, rule_name, e)
                for event in events:
                    if event["rule_name"] == rule_name:
                        for a in event["actions"]:
                            if a["kind"] == "run" and a["param"] == cmd and a["status"] == "pending":
                                a["status"] = run_status

            # Finalize overall event status and log history (after all statuses are final)
            for event in events:
                event["status"] = (
                    "applied"
                    if any(a["status"] == "applied" for a in event["actions"])
                    else "skipped"
                )
                self._log_history(event)

            # Attach events to supergraph for D4 feedback
            if events:
                self._gs._last_evolution_events.extend(
                    {"rule": e["rule_name"], "actions": e["actions"]} for e in events
                )

        finally:
            self._evaluating = False

        return events

    # -----------------------------------------------------------------------
    # History
    # -----------------------------------------------------------------------

    def history(self, limit: int = 10) -> list[dict]:
        if self._conn is None:
            return []
        rows = self._conn.execute(
            "SELECT rule_name, signals_json, actions_json, prev_values_json, status, timestamp "
            "FROM evolution_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        result = []
        for row in rows:
            result.append({
                "rule_name": row[0],
                "signals": json.loads(row[1]),
                "actions": json.loads(row[2]),
                "prev_values": json.loads(row[3]),
                "status": row[4],
                "timestamp": row[5],
            })
        return result

    # -----------------------------------------------------------------------
    # Persistence helpers
    # -----------------------------------------------------------------------

    def _persist_rule(self, rule: EvolutionRule) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO evolution_rules (name, rule_json, created_at) VALUES (?, ?, ?)",
            (rule.name, json.dumps(rule.to_dict()), time.time()),
        )
        self._conn.commit()

    def _load_from_db(self) -> None:
        if self._conn is None:
            return
        try:
            rows = self._conn.execute(
                "SELECT rule_json FROM evolution_rules"
            ).fetchall()
            for (rule_json,) in rows:
                d = json.loads(rule_json)
                rule = EvolutionRule.from_dict(d)
                self._rules[rule.name] = rule
        except Exception as e:
            logger.warning("Evolution: failed to load rules from DB: %s", e)

    def _log_history(self, event: dict) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "INSERT INTO evolution_history "
                "(timestamp, rule_name, signals_json, actions_json, prev_values_json, status) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    event["rule_name"],
                    json.dumps(event.get("signals", {}), default=str),
                    json.dumps(event.get("actions", []), default=str),
                    json.dumps(event.get("prev_values", {}), default=str),
                    event.get("status", "applied"),
                ),
            )
            self._conn.commit()
            self._prune_history()
        except Exception as e:
            logger.warning("Evolution: failed to log history: %s", e)

    def _prune_history(self) -> None:
        """Keep only the last history_retention entries."""
        try:
            self._conn.execute(
                "DELETE FROM evolution_history WHERE id NOT IN "
                "(SELECT id FROM evolution_history ORDER BY id DESC LIMIT ?)",
                (self._config.history_retention,),
            )
        except Exception:
            pass
