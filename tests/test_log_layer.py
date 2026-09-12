"""Tests for the intelligent log layer: auto-tagging, trace binding, SYS LOG."""
import logging
import pytest
from supergraph import SuperGraph
from supergraph.dsl.tagger import infer_tag, infer_phase
from supergraph.dsl.parser import parse


class TestAutoTagger:
    def test_read_query_tagged(self):
        ast = parse('NODES WHERE kind = "test"')
        assert infer_tag(ast) == "read"
        assert infer_phase("read") == "query"

    def test_write_query_tagged(self):
        ast = parse('CREATE NODE "x" kind = "test"')
        assert infer_tag(ast) == "write"
        assert infer_phase("write") == "mutation"

    def test_intelligence_query_tagged(self):
        ast = parse('SIMILAR TO "hello" LIMIT 5')
        assert infer_tag(ast) == "intelligence"

    def test_belief_tagged(self):
        ast = parse('ASSERT "fact1" kind = "fact" claim = "sky is blue"')
        assert infer_tag(ast) == "belief"

    def test_system_query_tagged(self):
        ast = parse('SYS STATS')
        assert infer_tag(ast) == "system"

    def test_vault_tagged(self):
        ast = parse('VAULT LIST')
        assert infer_tag(ast) == "vault"

    def test_ingest_tagged(self):
        ast = parse('INGEST "/tmp/test.txt"')
        assert infer_tag(ast) == "ingest"


class TestLogEnrichment:
    def test_log_entries_have_tags(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"))
        gs.execute('CREATE NODE "a" kind = "test" name = "Alice"')
        gs.execute('NODE "a"')

        result = gs.execute('SYS LOG LIMIT 10')
        assert result.kind == "log_entries"
        entries = result.data
        assert len(entries) >= 2

        write_entry = next(e for e in entries if "CREATE" in e["query"])
        assert write_entry["tag"] == "write"
        assert write_entry["phase"] == "mutation"
        assert write_entry["source"] == "user"

        read_entry = next(e for e in entries if 'NODE "a"' in e["query"] and e["tag"] == "read")
        assert read_entry["tag"] == "read"
        assert read_entry["phase"] == "query"
        assert read_entry["source"] == "user"

        gs.close()

    def test_trace_binding(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"))
        gs.bind_trace("session-42")
        gs.execute('CREATE NODE "traced" kind = "test"')
        gs.discard_trace()
        gs.execute('CREATE NODE "untraced" kind = "test"')

        result = gs.execute('SYS LOG TRACE "session-42"')
        entries = result.data
        assert len(entries) >= 1
        assert all(e["trace_id"] == "session-42" for e in entries)

        result2 = gs.execute('SYS LOG LIMIT 50')
        untraced = [e for e in result2.data if "untraced" in e.get("query", "")]
        if untraced:
            assert untraced[0]["trace_id"] is None

        gs.close()

    def test_sys_log_since_filter(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"))
        gs.execute('CREATE NODE "x" kind = "test"')
        result = gs.execute('SYS LOG SINCE "2020-01-01T00:00:00" LIMIT 10')
        assert result.kind == "log_entries"
        assert len(result.data) >= 1
        gs.close()

    def test_sys_log_empty_db(self):
        gs = SuperGraph()  # no persistence - no query log
        result = gs.execute('SYS LOG LIMIT 10')
        assert result.kind == "log_entries"
        assert result.data == []
        gs.close()


class TestEventLogger:
    def test_event_emitted(self, tmp_path, caplog):
        gs = SuperGraph(path=str(tmp_path / "db"))
        with caplog.at_level(logging.INFO, logger="supergraph.events"):
            gs.execute('CREATE NODE "evt" kind = "test"')
        assert any("write" in r.message for r in caplog.records), \
            f"Expected event log with 'write', got: {[r.message for r in caplog.records]}"
        gs.close()
