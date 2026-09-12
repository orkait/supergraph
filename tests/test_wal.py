"""Tests for WAL replay and query log rotation."""
import pytest
from supergraph import SuperGraph


def test_wal_replay_tolerates_duplicate_create(tmp_path):
    path = tmp_path / "gs"
    gs = SuperGraph(path=str(path))
    try:
        gs.execute('CREATE NODE "dup" kind = "doc" text = "x"')
        gs.checkpoint()
        gs._conn.execute(
            "INSERT INTO wal (timestamp, statement) VALUES (?, ?)",
            (0.0, 'CREATE NODE "dup" kind = "doc" text = "x"'),
        )
        gs._conn.commit()
    finally:
        # Simulate crash: skip checkpoint (so the forged WAL row persists)
        # but still drop the path lock so we can reopen.
        gs._wal = None
        if gs._conn is not None:
            gs._conn.close()
            gs._runtime.conn = None
        if gs._path_lock is not None:
            gs._path_lock.release()
            gs._path_lock = None

    gs2 = SuperGraph(path=str(path))
    try:
        assert gs2.execute('NODE "dup"').data is not None
    finally:
        gs2.close()


def test_query_log_row_cap(tmp_path):
    path = tmp_path / "gs"
    gs = SuperGraph(path=str(path))
    try:
        gs._wal._query_log_max_rows = 50
        for i in range(120):
            gs.execute('COUNT NODES')
        gs._wal.maybe_auto_checkpoint()
        count = gs._conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
        assert count <= 50
    finally:
        gs.close()


def test_wal_replay_moves_failing_statement_to_dlq(tmp_path):
    """A WAL statement that crashes replay lands in failed_wal_entries and
    gets removed from the main wal table so it does not loop forever."""
    path = tmp_path / "gs"
    gs = SuperGraph(path=str(path))
    try:
        gs.execute('CREATE NODE "ok" kind = "doc" text = "x"')
        gs.checkpoint()
        # Forge a statement that will fail replay (references a type the
        # schema does not know once a strict schema is registered later).
        gs._conn.execute(
            "INSERT INTO wal (timestamp, statement) VALUES (?, ?)",
            (0.0, 'NOT A VALID DSL STATEMENT AT ALL'),
        )
        gs._conn.commit()
    finally:
        # Simulate crash: skip checkpoint (so the forged WAL row persists)
        # but still drop the path lock so we can reopen.
        gs._wal = None
        if gs._conn is not None:
            gs._conn.close()
            gs._runtime.conn = None
        if gs._path_lock is not None:
            gs._path_lock.release()
            gs._path_lock = None

    gs2 = SuperGraph(path=str(path))
    try:
        # Failing entry should have been moved to DLQ, wal should not still
        # contain it (otherwise next replay would hit it again).
        dlq = gs2._conn.execute(
            "SELECT COUNT(*) FROM failed_wal_entries"
        ).fetchone()[0]
        assert dlq >= 1, "failing statement should be recorded in DLQ"
        wal_remaining = gs2._conn.execute(
            "SELECT COUNT(*) FROM wal WHERE statement = ?",
            ("NOT A VALID DSL STATEMENT AT ALL",),
        ).fetchone()[0]
        assert wal_remaining == 0, "failing statement must not stay in wal"
    finally:
        gs2.close()


def test_wal_replay_dlq_insert_failure_does_not_wedge_wal(tmp_path, monkeypatch):
    """If the DLQ insert itself fails, the main wal entry must still get
    deleted so replay does not infinite-loop on next open."""
    path = tmp_path / "gs"
    gs = SuperGraph(path=str(path))
    try:
        gs.execute('CREATE NODE "ok" kind = "doc" text = "x"')
        gs.checkpoint()
        gs._conn.execute(
            "INSERT INTO wal (timestamp, statement) VALUES (?, ?)",
            (0.0, 'NOT A VALID DSL STATEMENT AT ALL'),
        )
        gs._conn.commit()
        # Break the DLQ table so the insert raises.
        gs._conn.execute("DROP TABLE failed_wal_entries")
        gs._conn.commit()
    finally:
        # Simulate crash: skip checkpoint (so the forged WAL row persists)
        # but still drop the path lock so we can reopen.
        gs._wal = None
        if gs._conn is not None:
            gs._conn.close()
            gs._runtime.conn = None
        if gs._path_lock is not None:
            gs._path_lock.release()
            gs._path_lock = None

    gs2 = SuperGraph(path=str(path))
    try:
        # Even with DLQ broken, the bad statement should not remain in wal.
        wal_remaining = gs2._conn.execute(
            "SELECT COUNT(*) FROM wal WHERE statement = ?",
            ("NOT A VALID DSL STATEMENT AT ALL",),
        ).fetchone()[0]
        assert wal_remaining == 0, "bad statement must be deleted even if DLQ write failed"
    finally:
        gs2.close()
