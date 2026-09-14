from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from superclaw.dsl import MS_PER_DAY, Store, age, lit, now_ms, rows
from superclaw.session import NAMESPACE
from superclaw.settings import FACT_KIND, LIMITS, MAINT_KIND, MAINT_NODE
from supergraph.core.errors import SuperGraphError


@dataclass
class Report:
    expired: int = 0
    decayed: int = 0
    retracted: int = 0
    optimized: dict[str, Any] = field(default_factory=dict)
    health: dict[str, Any] = field(default_factory=dict)

    def line(self, dot: str) -> str:
        parts = [f"expired {self.expired}", f"decayed {self.decayed}", f"retracted {self.retracted}"]
        if self.optimized:
            parts.append("optimized " + ", ".join(f"{k} {v}" for k, v in self.optimized.items() if v))
        return f" {dot} ".join(parts)


def expire(gs: Store) -> int:
    return int((gs.execute("SYS EXPIRE").data or {}).get("expired", 0))


def decay(gs: Store) -> tuple[int, int]:
    cutoff = now_ms() - LIMITS.fact_decay_days * MS_PER_DAY
    decayed = retracted = 0
    query = f"NODES WHERE kind = {lit(FACT_KIND)} AND observed_at < NOW() - {LIMITS.fact_decay_days}d LIMIT {LIMITS.maintain_batch}"
    for row in rows(gs.execute(query, namespace=NAMESPACE)):
        if int(row.get("decayed_at") or 0) >= cutoff:
            continue
        confidence = float(row.get("confidence") or 0) * LIMITS.fact_decay_factor
        node = lit(str(row["id"]))
        if confidence < LIMITS.fact_confidence_floor:
            gs.execute(f'RETRACT {node} REASON "decayed below the confidence floor without a fresh assert"', namespace=NAMESPACE)
            retracted += 1
        else:
            gs.execute(f"UPDATE NODE {node} SET confidence = {confidence:.3f} decayed_at = {now_ms()}", namespace=NAMESPACE)
            decayed += 1
    return decayed, retracted


def health(gs: Store) -> dict[str, Any]:
    return dict(gs.execute("SYS HEALTH").data or {})


def maintain(gs: Store, *, optimize: bool = True) -> Report:
    report = Report(expired=expire(gs))
    report.decayed, report.retracted = decay(gs)
    if optimize:
        data = gs.execute("SYS OPTIMIZE").data or {}
        report.optimized = {name: sum(v for v in values.values() if isinstance(v, int)) for name, values in data.items() if isinstance(values, dict)}
    report.health = health(gs)
    gs.execute(
        f"UPSERT NODE {lit(MAINT_NODE)} kind = {lit(MAINT_KIND)} at = {now_ms()} expired = {report.expired} decayed = {report.decayed} retracted = {report.retracted}",
        namespace=NAMESPACE,
    )
    return report


def last(gs: Store) -> dict[str, Any] | None:
    try:
        data = gs.execute(f"NODE {lit(MAINT_NODE)}", namespace=NAMESPACE).data
    except SuperGraphError:
        return None
    return dict(data) if data else None


def stale(gs: Store) -> bool:
    previous = last(gs)
    return previous is None or int(previous.get("at") or 0) < now_ms() - LIMITS.maintain_stale_days * MS_PER_DAY


def health_line(gs: Store, dot: str) -> str:
    if gs is None:
        return "health not opened by this command"
    metrics = health(gs)
    previous = last(gs)
    when = age(int(previous["at"])) if previous and previous.get("at") else "never"
    return (f"health tombstones {float(metrics.get('tombstone_ratio') or 0):.1%} {dot} string bloat {float(metrics.get('string_bloat') or 0):.1f} "
            f"{dot} dead vectors {int(metrics.get('dead_vectors') or 0)} {dot} last maintain {when}")


def snapshots(gs: Store) -> list[str]:
    return [str(name) for name in (gs.execute("SYS SNAPSHOTS").data or [])]


def snapshot(gs: Store, name: str) -> bool:
    if name in snapshots(gs):
        return False
    gs.execute(f"SYS SNAPSHOT {lit(name)}")
    return True


def rollback(gs: Store, name: str) -> None:
    gs.execute(f"SYS ROLLBACK TO {lit(name)}")
