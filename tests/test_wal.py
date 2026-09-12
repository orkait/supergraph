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
    finally:
        gs._wal = None
        if gs._conn is not None:
            gs._conn.close()
            gs._runtime.conn = None
        if gs._path_lock is not None:
            gs._path_lock.release()
            gs._path_lock = None

    gs2 = SuperGraph(path=str(path))
    try:
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
        gs._conn.execute("DROP TABLE failed_wal_entries")
        gs._conn.commit()
    finally:
        gs._wal = None
        if gs._conn is not None:
            gs._conn.close()
            gs._runtime.conn = None
        if gs._path_lock is not None:
            gs._path_lock.release()
            gs._path_lock = None

    gs2 = SuperGraph(path=str(path))
    try:
        wal_remaining = gs2._conn.execute(
            "SELECT COUNT(*) FROM wal WHERE statement = ?",
            ("NOT A VALID DSL STATEMENT AT ALL",),
        ).fetchone()[0]
        assert wal_remaining == 0, "bad statement must be deleted even if DLQ write failed"
    finally:
        gs2.close()
