from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from croniter import croniter

from superclaw.app import Callbacks, Runtime, run_once
from superclaw.session import NAMESPACE, _lit
from superclaw.settings import LIMITS

KIND = "cronjob"
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_MS = 1000


class CronError(ValueError):
    pass


@dataclass(frozen=True)
class Job:
    id: str
    expr: str
    prompt: str
    model: str
    status: str
    fire_count: int
    next_run_ms: int
    created_ms: int
    last_exit: int

    @property
    def paused(self) -> bool:
        return self.status == "paused"


def _next_ms(expr: str, after_ms: int) -> int:
    base = datetime.fromtimestamp(after_ms / _MS, tz=timezone.utc)
    return int(croniter(expr, base).get_next(datetime).timestamp() * _MS)


def _job(row: dict[str, Any]) -> Job:
    return Job(id=str(row["jid"]), expr=str(row["expr"]), prompt=str(row.get("prompt", "")), model=str(row.get("model", "")),
               status=str(row.get("status", "active")), fire_count=int(row.get("fire_count", 0)), next_run_ms=int(row.get("next_run", 0)),
               created_ms=int(row.get("created", 0)), last_exit=int(row.get("last_exit", 0)))


class CronStore:
    def __init__(self, gs: Any, now_ms: Callable[[], int] = lambda: int(time.time() * _MS)) -> None:
        self._gs = gs
        self.now_ms = now_ms

    def _x(self, query: str):
        return self._gs.execute(query, namespace=NAMESPACE)

    def _node(self, job_id: str) -> str:
        return _lit(f"{KIND}:{job_id}")

    def add(self, job_id: str, expr: str, prompt: str, model: str = "") -> Job:
        job_id = job_id.strip().lower()
        if not _ID.match(job_id) or len(job_id) > LIMITS.cron_id_chars:
            raise CronError(f"invalid job id {job_id!r}: letters, digits, dots, dashes and underscores, up to {LIMITS.cron_id_chars} chars")
        if not croniter.is_valid(expr):
            raise CronError(f"invalid cron expression {expr!r}: five fields, or @hourly, @daily, @weekly, @monthly")
        if not prompt.strip():
            raise CronError("a prompt is required")
        if self.get(job_id) is not None:
            raise CronError(f"job {job_id!r} already exists; rm it first")
        now = self.now_ms()
        self._x(f'CREATE NODE {self._node(job_id)} kind = "{KIND}" jid = {_lit(job_id)} expr = {_lit(expr)} status = "active" '
                f'model = {_lit(model)} fire_count = 0 next_run = {_next_ms(expr, now)} created = {now} last_exit = 0 DOCUMENT {_lit(prompt.strip())}')
        return self.get(job_id)  # type: ignore[return-value]

    def get(self, job_id: str) -> Job | None:
        try:
            row = self._x(f"NODE {self._node(job_id)} WITH DOCUMENT").data
        except Exception:
            return None
        if not row:
            return None
        return _job({**row, "prompt": row.get("_document") or ""})

    def list(self) -> list[Job]:
        rows = self._x(f'NODES WHERE kind = "{KIND}" LIMIT {LIMITS.session_list_limit}').data or []
        jobs = [self.get(str(r["jid"])) for r in rows]
        return sorted((j for j in jobs if j), key=lambda j: j.id)

    def set_status(self, job_id: str, status: str) -> Job:
        job = self.get(job_id)
        if job is None:
            raise CronError(f"no job {job_id!r}")
        next_run = _next_ms(job.expr, self.now_ms()) if status == "active" and job.paused else job.next_run_ms
        self._x(f"UPDATE NODE {self._node(job_id)} SET status = {_lit(status)} next_run = {next_run}")
        return self.get(job_id)  # type: ignore[return-value]

    def remove(self, job_id: str) -> None:
        if self.get(job_id) is None:
            raise CronError(f"no job {job_id!r}")
        self._x(f"DELETE NODE {self._node(job_id)}")

    def due(self, now_ms: int | None = None) -> list[Job]:
        now = self.now_ms() if now_ms is None else now_ms
        return [j for j in self.list() if not j.paused and j.next_run_ms <= now]

    def reschedule(self, job_id: str, fired: bool, exit_code: int = 0) -> Job:
        job = self.get(job_id)
        if job is None:
            raise CronError(f"no job {job_id!r}")
        next_run = _next_ms(job.expr, self.now_ms())
        fires = job.fire_count + (1 if fired else 0)
        self._x(f"UPDATE NODE {self._node(job_id)} SET next_run = {next_run} fire_count = {fires} last_exit = {exit_code}")
        return self.get(job_id)  # type: ignore[return-value]


def fire(rt: Runtime, store: CronStore, job: Job, emit: Callable[[dict[str, Any]], None] = lambda event: None) -> int:
    sid = rt.store.create(cwd=str(rt.workspace), model=job.model or rt.model, title=f"cron {job.id}")
    try:
        result = run_once(rt, job.prompt, sid, Callbacks(on_event=emit))
        code = 2 if result.incomplete else 0
    except Exception as e:
        emit({"type": "error", "message": f"{type(e).__name__}: {e}", "job": job.id})
        code = 1
    store.reschedule(job.id, fired=True, exit_code=code)
    return code


def run(rt: Runtime, store: CronStore, ids: tuple[str, ...] = (), once: bool = False, catch_up: bool = False,
        emit: Callable[[dict[str, Any]], None] = lambda event: None, sleep: Callable[[float], None] = time.sleep,
        stop: Callable[[], bool] = lambda: False) -> int:
    selected = lambda job: not ids or job.id in ids  # noqa: E731
    if not once and not catch_up:
        for job in store.due():
            if selected(job):
                store.reschedule(job.id, fired=False)
                emit({"type": "cron_skipped", "job": job.id, "reason": "overdue at startup; pass --catch-up to run it now"})
    fired = 0
    while True:
        for job in store.due():
            if selected(job):
                emit({"type": "cron_fire", "job": job.id, "expr": job.expr})
                fire(rt, store, job, emit)
                fired += 1
        if once or stop():
            return fired
        pending = [j.next_run_ms for j in store.list() if not j.paused and selected(j)]
        if not pending:
            return fired
        sleep(max(0.0, min((min(pending) - store.now_ms()) / _MS, LIMITS.cron_tick_s)))
