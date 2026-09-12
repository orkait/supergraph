"""Tests for vault module: parser, manager, sync."""
import pytest

from supergraph.vault.parser import (
    parse_frontmatter, parse_sections, extract_wikilinks,
    title_to_slug, write_frontmatter, write_section,
)
from supergraph.vault.manager import VaultManager


SAMPLE_NOTE = """---
kind: memory
tags: [research, ai]
created: 2026-03-21T10:00:00
updated: 2026-03-21T10:00:00
status: active
agent: claude
---

## Summary
This is a summary about AI research.

## Body
Detailed content here.

## Links
- [[related-note]]
- [[another-note]]
"""


class TestParser:
    def test_parse_frontmatter(self):
        fm = parse_frontmatter(SAMPLE_NOTE)
        assert fm["kind"] == "memory"
        assert fm["status"] == "active"
        assert "research" in fm["tags"]

    def test_parse_frontmatter_empty(self):
        assert parse_frontmatter("no frontmatter here") == {}

    def test_parse_sections(self):
        sections = parse_sections(SAMPLE_NOTE)
        assert "summary" in sections
        assert "body" in sections
        assert "links" in sections
        assert "AI research" in sections["summary"]

    def test_extract_wikilinks(self):
        links = extract_wikilinks(SAMPLE_NOTE)
        assert "related-note" in links
        assert "another-note" in links

    def test_title_to_slug(self):
        assert title_to_slug("My Cool Note!") == "my-cool-note"
        assert title_to_slug("2026-03-21") == "2026-03-21"
        assert title_to_slug("  spaces  ") == "spaces"

    def test_write_frontmatter(self):
        updated = write_frontmatter(SAMPLE_NOTE, {"status": "archived"})
        fm = parse_frontmatter(updated)
        assert fm["status"] == "archived"
        assert fm["kind"] == "memory"  # preserved

    def test_write_section(self):
        updated = write_section(SAMPLE_NOTE, "body", "New body content")
        sections = parse_sections(updated)
        assert sections["body"] == "New body content"
        assert "AI research" in sections["summary"]  # other sections preserved

    def test_write_section_nonexistent_appends(self):
        updated = write_section(SAMPLE_NOTE, "instructions", "Do this thing")
        sections = parse_sections(updated)
        assert "instructions" in sections
        assert "Do this thing" in sections["instructions"]


class TestManager:
    @pytest.fixture
    def vault(self, tmp_path):
        return VaultManager(tmp_path / "notes")

    def test_new_creates_file(self, vault):
        slug = vault.new("My Research Note", kind="memory", tags=["ai", "ml"])
        assert slug == "my-research-note"
        assert (vault.path / "my-research-note.md").exists()

    def test_new_duplicate_raises(self, vault):
        vault.new("Test Note")
        with pytest.raises(FileExistsError):
            vault.new("Test Note")

    def test_read(self, vault):
        vault.new("Read Test", body="Hello world")
        content = vault.read("Read Test")
        assert "Hello world" in content
        assert "## Summary" in content

    def test_read_not_found(self, vault):
        with pytest.raises(FileNotFoundError):
            vault.read("nonexistent")

    def test_write_section(self, vault):
        vault.new("Write Test", body="old body")
        vault.write_section("Write Test", "body", "new body")
        content = vault.read("Write Test")
        assert "new body" in content
        assert "old body" not in content

    def test_append_section(self, vault):
        vault.new("Append Test", body="line 1")
        vault.append_section("Append Test", "body", "line 2")
        content = vault.read("Append Test")
        assert "line 1" in content
        assert "line 2" in content

    def test_daily(self, vault):
        slug = vault.daily()
        import datetime
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        assert slug == today
        assert (vault.path / f"{today}.md").exists()

    def test_daily_idempotent(self, vault):
        slug1 = vault.daily()
        slug2 = vault.daily()
        assert slug1 == slug2

    def test_archive(self, vault):
        vault.new("Archive Me")
        vault.archive("Archive Me")
        content = vault.read("Archive Me")
        fm = parse_frontmatter(content)
        assert fm["status"] == "archived"

    def test_list_files(self, vault):
        vault.new("Note A")
        vault.new("Note B")
        files = vault.list_files()
        assert "note-a" in files
        assert "note-b" in files


class TestVaultSync:
    def test_sync_creates_graph_nodes(self, tmp_path):
        from supergraph import SuperGraph
        vault_path = tmp_path / "notes"
        vault_path.mkdir()

        # Write a note file manually
        (vault_path / "test-note.md").write_text("""---
kind: memory
tags: [test]
created: 2026-03-21T10:00:00
updated: 2026-03-21T10:00:00
status: active
---

## Summary
A test note for sync.

## Body
Content here.

## Links
""")

        g = SuperGraph(path=str(tmp_path / "db"), vault=str(vault_path), embedder=None)
        # Sync should have run on init
        node = g.execute('NODE "note:test-note"')
        assert node.data is not None
        assert node.data["note_kind"] == "memory"
        assert node.data["status"] == "active"
        g.close()

    def test_sync_creates_wikilink_edges(self, tmp_path):
        from supergraph import SuperGraph
        vault_path = tmp_path / "notes"
        vault_path.mkdir()

        (vault_path / "note-a.md").write_text("""---
kind: memory
status: active
---

## Summary
Note A

## Links
- [[note-b]]
""")
        (vault_path / "note-b.md").write_text("""---
kind: memory
status: active
---

## Summary
Note B
""")

        g = SuperGraph(path=str(tmp_path / "db"), vault=str(vault_path), embedder=None)
        edges = g.execute('EDGES FROM "note:note-a"')
        link_edges = [e for e in edges.data if e["kind"] == "links"]
        assert len(link_edges) == 1
        assert link_edges[0]["target"] == "note:note-b"
        g.close()

    def test_vault_sync_api(self, tmp_path):
        """Test sync via Python API (VAULT SYNC DSL not yet wired)."""
        from supergraph import SuperGraph
        vault_path = tmp_path / "notes"
        vault_path.mkdir()
        (vault_path / "sync-test.md").write_text("""---
kind: fact
status: active
---

## Summary
Sync test note
""")
        g = SuperGraph(path=str(tmp_path / "db"), vault=str(vault_path), embedder=None)
        # Re-sync via Python API
        result = g._vault_sync.sync_all()
        assert result["synced"] >= 0
        assert result["errors"] == 0
        g.close()

    def test_sync_skips_unchanged(self, tmp_path):
        """Second sync_all should skip already-synced notes."""
        from supergraph import SuperGraph
        vault_path = tmp_path / "notes"
        vault_path.mkdir()
        (vault_path / "stable-note.md").write_text("""---
kind: memory
status: active
---

## Summary
Stable note
""")
        g = SuperGraph(path=str(tmp_path / "db"), vault=str(vault_path), embedder=None)
        # First sync happened in __init__, second should skip
        result = g._vault_sync.sync_all()
        assert result["skipped"] >= 1
        assert result["synced"] == 0
        g.close()


class TestFactAutoAssert:
    def test_fact_note_sets_confidence(self, tmp_path):
        from supergraph import SuperGraph
        vault_path = tmp_path / "notes"
        vault_path.mkdir()
        (vault_path / "db-version.md").write_text("""---
kind: fact
confidence: 0.95
source: human
status: active
---

## Summary
Production database is PostgreSQL 15.3

## Body
Verified on 2026-03-21.
""")
        g = SuperGraph(path=str(tmp_path / "db"), vault=str(vault_path), embedder=None)
        node = g.execute('NODE "note:db-version"')
        assert node.data is not None
        assert node.data["note_kind"] == "fact"
        # Check that belief columns were set via reserved columns
        slot = 0
        cols = g._store.columns
        assert cols.has_column("__confidence__")
        assert cols._presence["__confidence__"][slot]
        assert abs(float(cols._columns["__confidence__"][slot]) - 0.95) < 0.01
        assert cols.has_column("__source__")
        assert cols._presence["__source__"][slot]
        g.close()

    def test_non_fact_note_has_no_confidence(self, tmp_path):
        from supergraph import SuperGraph
        vault_path = tmp_path / "notes"
        vault_path.mkdir()
        (vault_path / "regular.md").write_text("""---
kind: memory
status: active
---

## Summary
Just a regular memory.
""")
        g = SuperGraph(path=str(tmp_path / "db"), vault=str(vault_path), embedder=None)
        node = g.execute('NODE "note:regular"')
        assert node.data is not None
        # No __confidence__ set for non-fact notes
        slot = 0
        cols = g._store.columns
        if cols.has_column("__confidence__"):
            assert not cols._presence["__confidence__"][slot]
        g.close()


class TestVaultDSL:
    def test_vault_new(self, tmp_path):
        from supergraph import SuperGraph
        g = SuperGraph(path=str(tmp_path / "db"), vault=str(tmp_path / "notes"), embedder=None)
        result = g.execute('VAULT NEW "My Research" KIND "memory" TAGS "ai,ml"')
        assert result.data["slug"] == "my-research"
        assert (tmp_path / "notes" / "my-research.md").exists()
        # Node should exist in graph
        node = g.execute('NODE "note:my-research"')
        assert node.data is not None
        assert node.data["note_kind"] == "memory"
        g.close()

    def test_vault_read(self, tmp_path):
        from supergraph import SuperGraph
        g = SuperGraph(path=str(tmp_path / "db"), vault=str(tmp_path / "notes"), embedder=None)
        g.execute('VAULT NEW "Read Test" KIND "fact"')
        result = g.execute('VAULT READ "Read Test"')
        assert result.kind == "note"
        assert "## Summary" in result.data["content"]
        g.close()

    def test_vault_write_section(self, tmp_path):
        from supergraph import SuperGraph
        g = SuperGraph(path=str(tmp_path / "db"), vault=str(tmp_path / "notes"), embedder=None)
        g.execute('VAULT NEW "Write Test"')
        g.execute('VAULT WRITE "Write Test" SECTION "body" CONTENT "Updated body content"')
        result = g.execute('VAULT READ "Write Test"')
        assert "Updated body content" in result.data["content"]
        g.close()

    def test_vault_append_section(self, tmp_path):
        from supergraph import SuperGraph
        g = SuperGraph(path=str(tmp_path / "db"), vault=str(tmp_path / "notes"), embedder=None)
        g.execute('VAULT NEW "Append Test"')
        g.execute('VAULT APPEND "Append Test" SECTION "body" CONTENT "Line 1"')
        g.execute('VAULT APPEND "Append Test" SECTION "body" CONTENT "Line 2"')
        result = g.execute('VAULT READ "Append Test"')
        assert "Line 1" in result.data["content"]
        assert "Line 2" in result.data["content"]
        g.close()

    def test_vault_list(self, tmp_path):
        from supergraph import SuperGraph
        g = SuperGraph(path=str(tmp_path / "db"), vault=str(tmp_path / "notes"), embedder=None)
        g.execute('VAULT NEW "Note A" KIND "memory"')
        g.execute('VAULT NEW "Note B" KIND "instruction"')
        result = g.execute('VAULT LIST')
        assert result.count >= 2
        g.close()

    def test_vault_list_with_where(self, tmp_path):
        from supergraph import SuperGraph
        g = SuperGraph(path=str(tmp_path / "db"), vault=str(tmp_path / "notes"), embedder=None)
        g.execute('VAULT NEW "Memory Note" KIND "memory"')
        g.execute('VAULT NEW "Instruction Note" KIND "instruction"')
        result = g.execute('VAULT LIST WHERE note_kind = "instruction"')
        assert result.count == 1
        assert result.data[0]["note_kind"] == "instruction"
        g.close()

    def test_vault_sync(self, tmp_path):
        from supergraph import SuperGraph
        g = SuperGraph(path=str(tmp_path / "db"), vault=str(tmp_path / "notes"), embedder=None)
        result = g.execute('VAULT SYNC')
        assert "synced" in result.data
        g.close()

    def test_vault_daily(self, tmp_path):
        from supergraph import SuperGraph
        import datetime
        g = SuperGraph(path=str(tmp_path / "db"), vault=str(tmp_path / "notes"), embedder=None)
        result = g.execute('VAULT DAILY')
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        assert result.data["slug"] == today
        g.close()

    def test_vault_archive(self, tmp_path):
        from supergraph import SuperGraph
        g = SuperGraph(path=str(tmp_path / "db"), vault=str(tmp_path / "notes"), embedder=None)
        g.execute('VAULT NEW "Archive Me"')
        g.execute('VAULT ARCHIVE "Archive Me"')
        node = g.execute('NODE "note:archive-me"')
        assert node.data["status"] == "archived"
        g.close()

    def test_vault_search_fallback(self, tmp_path):
        from supergraph import SuperGraph
        g = SuperGraph(path=str(tmp_path / "db"), vault=str(tmp_path / "notes"), embedder=None)
        g.execute('VAULT NEW "AI Research" KIND "memory"')
        # Write a summary that's searchable
        g.execute('VAULT WRITE "AI Research" SECTION "summary" CONTENT "Deep learning transformer models"')
        result = g.execute('VAULT SEARCH "transformer" LIMIT 5')
        assert result.count >= 1
        g.close()

    def test_vault_without_vault_raises(self, tmp_path):
        import pytest as _pytest
        from supergraph import SuperGraph
        g = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        with _pytest.raises(Exception, match="[Vv]ault"):
            g.execute('VAULT NEW "test"')
        g.close()

    def test_vault_backlinks(self, tmp_path):
        from supergraph import SuperGraph
        vault_path = tmp_path / "notes"
        vault_path.mkdir()
        # Create notes with wikilinks
        (vault_path / "note-a.md").write_text("""---
kind: memory
status: active
---

## Summary
Note A

## Links
- [[note-b]]
""")
        (vault_path / "note-b.md").write_text("""---
kind: memory
status: active
---

## Summary
Note B
""")
        g = SuperGraph(path=str(tmp_path / "db"), vault=str(vault_path), embedder=None)
        result = g.execute('VAULT BACKLINKS "note-b"')
        assert result.count >= 1
        g.close()
