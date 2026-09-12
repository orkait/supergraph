"""Test that incremental checkpoint only writes dirty subsystems."""
import tempfile
import sqlite3
from pathlib import Path
from supergraph import SuperGraph


def test_incremental_checkpoint_skips_clean_data():
    """After a full checkpoint, a second checkpoint with no changes
    should not rewrite node/edge blobs (verified by comparing blob content)."""
    with tempfile.TemporaryDirectory() as td:
        gs = SuperGraph(path=td)
        gs.execute('CREATE NODE "a" kind = "test" name = "Alice"')
        gs.execute('CREATE NODE "b" kind = "test" name = "Bob"')
        gs.execute('CREATE EDGE "a" -> "b" kind = "knows"')
        gs.checkpoint()

        # Snapshot blob content after first checkpoint
        db_path = Path(td) / "supergraph.db"
        conn = sqlite3.connect(str(db_path))
        blobs_before = {
            row[0]: row[1]
            for row in conn.execute("SELECT key, data FROM blobs").fetchall()
        }
        # Insert a sentinel to detect if strings blob is rewritten
        conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                     ("_test_sentinel", b"marker", "test"))
        conn.commit()
        conn.close()

        # Second checkpoint with NO changes
        gs.checkpoint()

        # Verify sentinel survived (proves blobs table wasn't wiped)
        conn = sqlite3.connect(str(db_path))
        sentinel = conn.execute(
            "SELECT data FROM blobs WHERE key = '_test_sentinel'"
        ).fetchone()
        assert sentinel is not None, "Sentinel was deleted - checkpoint rewrote all blobs"

        # Verify node blobs are byte-identical (not rewritten needlessly)
        node_ids_after = conn.execute(
            "SELECT data FROM blobs WHERE key = 'node_ids'"
        ).fetchone()
        assert node_ids_after[0] == blobs_before["node_ids"], \
            "node_ids blob was rewritten despite no changes"
        conn.close()
        gs.close()


def test_incremental_checkpoint_writes_dirty_nodes():
    """After modifying a node, checkpoint writes node-related blobs."""
    with tempfile.TemporaryDirectory() as td:
        gs = SuperGraph(path=td)
        gs.execute('CREATE NODE "a" kind = "test" name = "Alice"')
        gs.checkpoint()

        # Modify a node
        gs.execute('UPDATE NODE "a" SET name = "Bob"')
        gs.checkpoint()

        # Verify the change persisted
        gs.close()
        gs2 = SuperGraph(path=td)
        result = gs2.execute('NODE "a"')
        assert result.data["name"] == "Bob"
        gs2.close()
