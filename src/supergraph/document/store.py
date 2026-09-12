"""DocumentStore: SQLite multi-table storage for documents, summaries, metadata, images."""

import sqlite3
import tempfile
import os
from pathlib import Path

from supergraph.algos.text import fts5_sanitize as _sanitize_fts5_query


_TEXT_CONTENT_TYPES = (
    "text/plain", "text/markdown", "text/html", "text/xml",
    "text/csv", "text/rtf",
    "application/json", "application/xml", "application/xhtml+xml",
)


def _is_text_content_type(content_type: str) -> bool:
    """True when content_type indicates UTF-8 decodable text.

    BM25 indexing of binary data (PDF bytes, images) ruins ranking, so
    put_document only auto-writes to doc_fts for known text content types.
    """
    if not content_type:
        return False
    ct = content_type.split(";", 1)[0].strip().lower()
    return ct in _TEXT_CONTENT_TYPES or ct.startswith("text/")


class DocumentStore:
    """SQLite-backed storage for raw documents. Always on disk, never in RAM."""

    def __init__(self, db_path: str | None = None, fts_tokenizer: str = "porter unicode61"):
        """Initialize with a file path. If None, uses a temp file (cleaned on close).

        ``check_same_thread=False`` so the connection works under uvicorn /
        FastAPI / multi-threaded callers. supergraph serializes writes via
        higher-level locking around the SuperGraph facade, and sqlite3
        itself uses its own internal mutex when this flag is set.
        """
        self._fts_tokenizer = fts_tokenizer
        if db_path:
            self._path = db_path
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._temp = False
        else:
            fd, self._path = tempfile.mkstemp(suffix=".supergraph-docs.db")
            os.close(fd)  # close fd, sqlite will reopen
            self._conn = sqlite3.connect(self._path, check_same_thread=False)
            self._temp = True
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_tables()

    def _ensure_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                slot INTEGER PRIMARY KEY,
                content BLOB NOT NULL,
                content_type TEXT NOT NULL,
                size INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS summaries (
                slot INTEGER PRIMARY KEY,
                summary TEXT NOT NULL,
                heading TEXT,
                page INTEGER,
                chunk_index INTEGER,
                doc_slot INTEGER
            );
            CREATE TABLE IF NOT EXISTS doc_metadata (
                doc_slot INTEGER PRIMARY KEY,
                source_path TEXT,
                pages INTEGER,
                author TEXT,
                title TEXT,
                parser_used TEXT,
                confidence REAL,
                ingested_at INTEGER
            );
            CREATE TABLE IF NOT EXISTS images (
                slot INTEGER PRIMARY KEY,
                image_data BLOB NOT NULL,
                mime_type TEXT NOT NULL,
                page INTEGER,
                description TEXT
            );
        """)
        self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS doc_fts
            USING fts5(summary, tokenize='{self._fts_tokenizer}')
        """)

    # --- Documents ---
    def put_document(self, slot: int, content: bytes, content_type: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO documents (slot, content, content_type, size) VALUES (?, ?, ?, ?)",
            (slot, content, content_type, len(content)))
        # Always clear any prior FTS row first. Without this, overwriting a
        # text document with a binary one left the old text indexed in doc_fts
        # pointing at the now-binary slot (bug #51). A subsequent BM25 query
        # would return the slot for content it no longer contains.
        self._conn.execute("DELETE FROM doc_fts WHERE rowid = ?", (slot,))
        # Plaintext documents also land in the BM25 index so REMEMBER can find
        # them without a separate put_summary() call. Binary/PDF content stays
        # out of FTS (indexing garbled bytes hurts ranking).
        if _is_text_content_type(content_type):
            try:
                text = content.decode("utf-8", errors="replace")
            except Exception:
                text = None
            if text:
                self._conn.execute(
                    "INSERT INTO doc_fts (rowid, summary) VALUES (?, ?)",
                    (slot, text))
        self._conn.commit()

    def put_documents_batch(self, rows: list[tuple[int, bytes, str]]) -> None:
        """Batch insert documents with single commit."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO documents (slot, content, content_type, size) VALUES (?, ?, ?, ?)",
            [(slot, content, ctype, len(content)) for slot, content, ctype in rows])
        # Same BM25 auto-index path as put_document, batched.
        fts_rows: list[tuple[int, str]] = []
        for slot, content, ctype in rows:
            if not _is_text_content_type(ctype):
                continue
            try:
                text = content.decode("utf-8", errors="replace")
            except Exception:
                continue
            if text:
                fts_rows.append((slot, text))
        if fts_rows:
            self._conn.executemany(
                "INSERT OR REPLACE INTO doc_fts (rowid, summary) VALUES (?, ?)",
                fts_rows)
        self._conn.commit()

    def get_document(self, slot: int) -> tuple[bytes, str] | None:
        row = self._conn.execute(
            "SELECT content, content_type FROM documents WHERE slot = ?", (slot,)).fetchone()
        return (row[0], row[1]) if row else None

    def delete_document(self, slot: int) -> None:
        self._conn.execute("DELETE FROM documents WHERE slot = ?", (slot,))
        # Mirror the auto-FTS in put_document so deletion is symmetric. Safe
        # to issue unconditionally - no-op if this slot never had an FTS row.
        self._conn.execute("DELETE FROM doc_fts WHERE rowid = ?", (slot,))
        self._conn.commit()

    def delete_documents_batch(self, slots: list[int]) -> None:
        """Batch delete documents with single commit."""
        self._conn.executemany("DELETE FROM documents WHERE slot = ?", [(s,) for s in slots])
        self._conn.executemany("DELETE FROM doc_fts WHERE rowid = ?", [(s,) for s in slots])
        self._conn.commit()

    def has_document(self, slot: int) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM documents WHERE slot = ?", (slot,)).fetchone() is not None

    # --- Summaries ---
    def put_summary(self, slot: int, summary: str, heading: str | None = None,
                    page: int | None = None, chunk_index: int = 0, doc_slot: int = 0) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO summaries VALUES (?, ?, ?, ?, ?, ?)",
            (slot, summary, heading, page, chunk_index, doc_slot))
        self._conn.execute(
            "INSERT OR REPLACE INTO doc_fts (rowid, summary) VALUES (?, ?)",
            (slot, summary))
        self._conn.commit()

    def put_summaries_batch(self, rows: list[tuple[int, str, str | None, int | None, int, int]]) -> None:
        """Batch insert summaries with single commit."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO summaries VALUES (?, ?, ?, ?, ?, ?)",
            rows)
        self._conn.executemany(
            "INSERT OR REPLACE INTO doc_fts (rowid, summary) VALUES (?, ?)",
            [(r[0], r[1]) for r in rows])
        self._conn.commit()

    def get_summary(self, slot: int) -> dict | None:
        row = self._conn.execute(
            "SELECT summary, heading, page, chunk_index, doc_slot FROM summaries WHERE slot = ?",
            (slot,)).fetchone()
        if not row:
            return None
        return {"summary": row[0], "heading": row[1], "page": row[2],
                "chunk_index": row[3], "doc_slot": row[4]}

    def get_summaries_for_doc(self, doc_slot: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT slot, summary, heading, page, chunk_index FROM summaries WHERE doc_slot = ?",
            (doc_slot,)).fetchall()
        return [{"slot": r[0], "summary": r[1], "heading": r[2], "page": r[3],
                 "chunk_index": r[4]} for r in rows]

    # --- Full-text search ---
    def search_text(self, query: str, limit: int = 10) -> list[tuple[int, float]]:
        """BM25 full-text search over summaries. Returns [(slot, bm25_score), ...]."""
        safe_query = _sanitize_fts5_query(query)
        if not safe_query:
            return []
        rows = self._conn.execute(
            "SELECT rowid, rank FROM doc_fts WHERE doc_fts MATCH ? ORDER BY rank LIMIT ?",
            (safe_query, limit)
        ).fetchall()
        return [(int(row[0]), float(row[1])) for row in rows]

    # --- Metadata ---
    def put_metadata(self, doc_slot: int, metadata: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO doc_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (doc_slot, metadata.get("source_path"), metadata.get("pages"),
             metadata.get("author"), metadata.get("title"),
             metadata.get("parser_used"), metadata.get("confidence"),
             metadata.get("ingested_at")))
        self._conn.commit()

    def get_metadata(self, doc_slot: int) -> dict | None:
        row = self._conn.execute(
            "SELECT source_path, pages, author, title, parser_used, confidence, ingested_at "
            "FROM doc_metadata WHERE doc_slot = ?", (doc_slot,)).fetchone()
        if not row:
            return None
        return {"source_path": row[0], "pages": row[1], "author": row[2],
                "title": row[3], "parser_used": row[4], "confidence": row[5],
                "ingested_at": row[6]}

    # --- Images ---
    def put_image(self, slot: int, image_data: bytes, mime_type: str,
                  page: int | None = None, description: str | None = None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO images VALUES (?, ?, ?, ?, ?)",
            (slot, image_data, mime_type, page, description))
        self._conn.commit()

    def get_image(self, slot: int) -> dict | None:
        row = self._conn.execute(
            "SELECT image_data, mime_type, page, description FROM images WHERE slot = ?",
            (slot,)).fetchone()
        if not row:
            return None
        return {"image_data": row[0], "mime_type": row[1], "page": row[2], "description": row[3]}

    # --- Bulk operations ---
    def delete_all_for_doc(self, doc_slot: int) -> int:
        """Delete document + all its summaries + images. Returns count deleted."""
        # Find chunk slots for this doc
        chunk_rows = self._conn.execute(
            "SELECT slot FROM summaries WHERE doc_slot = ?", (doc_slot,)).fetchall()
        chunk_slots = [r[0] for r in chunk_rows]

        count = 0
        # Delete document
        r = self._conn.execute("DELETE FROM documents WHERE slot = ?", (doc_slot,))
        count += r.rowcount
        # Delete chunk documents
        for s in chunk_slots:
            r = self._conn.execute("DELETE FROM documents WHERE slot = ?", (s,))
            count += r.rowcount
        # Delete summaries
        r = self._conn.execute("DELETE FROM summaries WHERE doc_slot = ?", (doc_slot,))
        count += r.rowcount
        # Delete metadata
        self._conn.execute("DELETE FROM doc_metadata WHERE doc_slot = ?", (doc_slot,))
        # Delete images (by doc_slot... but images table uses slot, not doc_slot)
        # Images are linked via graph edges, not doc_slot in images table
        # We delete images whose slots are in chunk_slots
        for s in chunk_slots:
            r = self._conn.execute("DELETE FROM images WHERE slot = ?", (s,))
            count += r.rowcount
        # Clean FTS index
        self._conn.execute("DELETE FROM doc_fts WHERE rowid = ?", (doc_slot,))
        for s in chunk_slots:
            self._conn.execute("DELETE FROM doc_fts WHERE rowid = ?", (s,))

        self._conn.commit()
        return count

    def orphan_cleanup(self, live_slots: set[int]) -> int:
        """Delete entries whose slots are not in live_slots. For rollback cleanup."""
        all_doc_slots = {r[0] for r in self._conn.execute("SELECT slot FROM documents").fetchall()}
        all_sum_slots = {r[0] for r in self._conn.execute("SELECT slot FROM summaries").fetchall()}
        all_img_slots = {r[0] for r in self._conn.execute("SELECT slot FROM images").fetchall()}

        orphan_docs = all_doc_slots - live_slots
        orphan_sums = all_sum_slots - live_slots
        orphan_imgs = all_img_slots - live_slots

        count = 0
        for s in orphan_docs:
            self._conn.execute("DELETE FROM documents WHERE slot = ?", (s,))
            count += 1
        for s in orphan_sums:
            self._conn.execute("DELETE FROM summaries WHERE slot = ?", (s,))
            count += 1
        for s in orphan_imgs:
            self._conn.execute("DELETE FROM images WHERE slot = ?", (s,))
            count += 1

        # Clean metadata for orphaned docs
        all_meta_slots = {r[0] for r in self._conn.execute("SELECT doc_slot FROM doc_metadata").fetchall()}
        for s in all_meta_slots - live_slots:
            self._conn.execute("DELETE FROM doc_metadata WHERE doc_slot = ?", (s,))

        # Clean FTS for orphaned summaries AND orphaned documents.
        # put_document auto-writes plaintext bodies to doc_fts (PR #102),
        # so orphaned documents can leave stale BM25 rows behind -
        # REMEMBER would then return matches pointing at tombstoned slots.
        for s in orphan_sums | orphan_docs:
            self._conn.execute("DELETE FROM doc_fts WHERE rowid = ?", (s,))

        self._conn.commit()
        return count

    def stats(self) -> dict:
        doc_count = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        total_bytes = self._conn.execute("SELECT COALESCE(SUM(size), 0) FROM documents").fetchone()[0]
        chunk_count = self._conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
        image_count = self._conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        return {
            "document_count": doc_count,
            "total_bytes": total_bytes,
            "chunk_count": chunk_count,
            "image_count": image_count,
        }

    def close(self):
        self._conn.close()
        if self._temp and os.path.exists(self._path):
            os.unlink(self._path)
