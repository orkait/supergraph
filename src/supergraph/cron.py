
import time
import threading
import logging
import sqlite3
from concurrent.futures import Future

logger = logging.getLogger(__name__)


class CronScheduler:

    def __init__(self, conn: sqlite3.Connection | None, submit_fn):
        self._conn = conn
        self._tick_conn: sqlite3.Connection | None = None
        self._tick_db_path: str | None = None
        self._submit = submit_fn
        self._thread: threading.Thread | None = None
        self._running = False
        if conn is not None:
            try:
                row = conn.execute("PRAGMA database_list").fetchone()
                if row and len(row) >= 3:
                    self._tick_db_path = row[2]
            except Exception as e:
                logger.debug("CronScheduler: could not resolve db path: %s", e)

    def start(self) -> None:
        if self._conn is None:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._tick_loop, daemon=True, name="supergraph-cron"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._tick_conn is not None:
            try:
                self._tick_conn.close()
            except Exception:
                pass
            self._tick_conn = None

    def add(self, name: str, schedule: str, query: str) -> dict:
        from croniter import croniter
        if not croniter.is_valid(schedule):
            raise ValueError(f"Invalid cron expression: {schedule!r}")
        now = time.time()
        next_run = croniter(schedule, now).get_next(float)
        conn = self._conn
        if conn is None:
            raise RuntimeError("CRON requires persistence (SuperGraph with path)")
        try:
            conn.execute(
                "INSERT INTO cron_jobs (name, schedule, query, enabled, created_at, next_run) "
                "VALUES (?, ?, ?, 1, ?, ?)",
                (name, schedule, query, now, next_run),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Cron job already exists: {name!r}")
        return {"name": name, "schedule": schedule, "query": query, "next_run": next_run}

    def delete(self, name: str) -> None:
        conn = self._conn
        if conn is None:
            return
        r = conn.execute("DELETE FROM cron_jobs WHERE name = ?", (name,))
        conn.commit()
        if r.rowcount == 0:
            raise ValueError(f"Cron job not found: {name!r}")

    def enable(self, name: str) -> None:
        self._set_enabled(name, 1)

    def disable(self, name: str) -> None:
        self._set_enabled(name, 0)

    def _set_enabled(self, name: str, value: int) -> None:
        conn = self._conn
        if conn is None:
            return
        r = conn.execute(
            "UPDATE cron_jobs SET enabled = ? WHERE name = ?", (value, name)
        )
        conn.commit()
        if r.rowcount == 0:
            raise ValueError(f"Cron job not found: {name!r}")

    def list_jobs(self) -> list[dict]:
        conn = self._conn
        if conn is None:
            return []
        rows = conn.execute(
            "SELECT name, schedule, query, enabled, created_at, last_run, next_run, "
            "run_count, error_count, last_error FROM cron_jobs ORDER BY name"
        ).fetchall()
        return [
            {
                "name": r[0], "schedule": r[1], "query": r[2],
                "enabled": bool(r[3]), "created_at": r[4],
                "last_run": r[5], "next_run": r[6],
                "run_count": r[7], "error_count": r[8], "last_error": r[9],
            }
            for r in rows
        ]

    def run_now(self, name: str) -> Future:
        conn = self._conn
        if conn is None:
            raise RuntimeError("CRON requires persistence")
        row = conn.execute(
            "SELECT query FROM cron_jobs WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Cron job not found: {name!r}")
        query = row[0]
        future = self._submit(query)
        future.add_done_callback(lambda f: self._on_done(f, name))
        return future

    def _tick_loop(self) -> None:
        if self._tick_db_path is not None:
            try:
                self._tick_conn = sqlite3.connect(
                    self._tick_db_path, check_same_thread=False,
                )
            except Exception as e:
                logger.warning(
                    "CronScheduler: failed to open dedicated connection to %s: %s",
                    self._tick_db_path, e,
                )
                self._tick_conn = None
        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.debug("cron tick error: %s", e)
            for _ in range(60):
                if not self._running:
                    return
                time.sleep(1)

    def _tick(self) -> None:
        conn = self._tick_conn if self._tick_conn is not None else self._conn
        if conn is None:
            return
        now = time.time()
        rows = conn.execute(
            "SELECT name, query, schedule FROM cron_jobs "
            "WHERE enabled = 1 AND next_run <= ?",
            (now,),
        ).fetchall()
        for name, query, schedule in rows:
            try:
                future = self._submit(query)
                future.add_done_callback(lambda f, n=name: self._on_done(f, n))
                from croniter import croniter
                next_run = croniter(schedule, now).get_next(float)
                conn.execute(
                    "UPDATE cron_jobs SET last_run = ?, next_run = ?, "
                    "run_count = run_count + 1 WHERE name = ?",
                    (now, next_run, name),
                )
                conn.commit()
            except Exception as e:
                logger.debug("cron job %s submission failed: %s", name, e)
                conn.execute(
                    "UPDATE cron_jobs SET error_count = error_count + 1, "
                    "last_error = ? WHERE name = ?",
                    (str(e), name),
                )
                conn.commit()

    def _on_done(self, future: Future, job_name: str) -> None:
        exc = future.exception()
        if exc is not None:
            logger.warning("cron job %s failed: %s", job_name, exc)
            conn = self._conn
            if conn is not None:
                try:
                    conn.execute(
                        "UPDATE cron_jobs SET error_count = error_count + 1, "
                        "last_error = ? WHERE name = ?",
                        (str(exc), job_name),
                    )
                    conn.commit()
                except Exception:
                    pass
