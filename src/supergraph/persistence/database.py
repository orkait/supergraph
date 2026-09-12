
import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1


def open_database(path: str | Path, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS blobs (
            key TEXT PRIMARY KEY,
            data BLOB,
            dtype TEXT
        );
        CREATE TABLE IF NOT EXISTS wal (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            statement TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS failed_wal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            statement TEXT NOT NULL,
            error_msg TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            query TEXT NOT NULL,
            elapsed_us INTEGER NOT NULL,
            result_count INTEGER NOT NULL,
            error TEXT,
            tag TEXT,
            trace_id TEXT,
            source TEXT DEFAULT 'user',
            phase TEXT
        );
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cron_jobs (
            name TEXT PRIMARY KEY,
            schedule TEXT NOT NULL,
            query TEXT NOT NULL,
            enabled INTEGER DEFAULT 1,
            created_at REAL NOT NULL,
            last_run REAL,
            next_run REAL NOT NULL,
            run_count INTEGER DEFAULT 0,
            error_count INTEGER DEFAULT 0,
            last_error TEXT
        );
        CREATE TABLE IF NOT EXISTS evolution_rules (
            name TEXT PRIMARY KEY,
            rule_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS evolution_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            rule_name TEXT NOT NULL,
            signals_json TEXT NOT NULL,
            actions_json TEXT NOT NULL,
            prev_values_json TEXT NOT NULL,
            status TEXT NOT NULL
        );
    """)
    _migrate_query_log(conn)
    conn.commit()


def _migrate_query_log(conn: sqlite3.Connection):
    cursor = conn.execute("PRAGMA table_info(query_log)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    migrations = [
        ("tag", "TEXT"),
        ("trace_id", "TEXT"),
        ("source", "TEXT DEFAULT 'user'"),
        ("phase", "TEXT"),
    ]
    for col_name, col_type in migrations:
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE query_log ADD COLUMN {col_name} {col_type}")


def get_metadata(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def set_metadata(conn: sqlite3.Connection, key: str, value: str):
    conn.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", (key, value))
    conn.commit()
