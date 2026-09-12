import tempfile
import sqlite3
from pathlib import Path
from supergraph import SuperGraph


def test_incremental_checkpoint_skips_clean_data():
    with tempfile.TemporaryDirectory() as td:
        gs = SuperGraph(path=td)
        gs.execute('CREATE NODE "a" kind = "test" name = "Alice"')
        gs.execute('CREATE NODE "b" kind = "test" name = "Bob"')
        gs.execute('CREATE EDGE "a" -> "b" kind = "knows"')
        gs.checkpoint()

        db_path = Path(td) / "supergraph.db"
        conn = sqlite3.connect(str(db_path))
        blobs_before = {
            row[0]: row[1]
            for row in conn.execute("SELECT key, data FROM blobs").fetchall()
        }
        conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                     ("_test_sentinel", b"marker", "test"))
        conn.commit()
        conn.close()

        gs.checkpoint()

        conn = sqlite3.connect(str(db_path))
        sentinel = conn.execute(
            "SELECT data FROM blobs WHERE key = '_test_sentinel'"
        ).fetchone()
        assert sentinel is not None, "Sentinel was deleted - checkpoint rewrote all blobs"

        node_ids_after = conn.execute(
            "SELECT data FROM blobs WHERE key = 'node_ids'"
        ).fetchone()
        assert node_ids_after[0] == blobs_before["node_ids"], \
            "node_ids blob was rewritten despite no changes"
        conn.close()
        gs.close()


def test_incremental_checkpoint_writes_dirty_nodes():
    with tempfile.TemporaryDirectory() as td:
        gs = SuperGraph(path=td)
        gs.execute('CREATE NODE "a" kind = "test" name = "Alice"')
        gs.checkpoint()

        gs.execute('UPDATE NODE "a" SET name = "Bob"')
        gs.checkpoint()

        gs.close()
        gs2 = SuperGraph(path=td)
        result = gs2.execute('NODE "a"')
        assert result.data["name"] == "Bob"
        gs2.close()
