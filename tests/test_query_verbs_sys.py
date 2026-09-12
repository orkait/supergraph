import pytest

from supergraph import q, F
from supergraph.dsl.parser import parse


def _roundtrip(query_obj):
    dsl = query_obj.dsl()
    try:
        parse(dsl)
    except Exception as e:
        pytest.fail(f"parser rejected {dsl!r}: {e}")
    return dsl


class TestSysScalar:
    def test_status(self):
        assert _roundtrip(q.sys.status()) == "SYS STATUS"

    def test_health(self):
        assert _roundtrip(q.sys.health()) == "SYS HEALTH"

    def test_retain(self):
        assert _roundtrip(q.sys.retain()) == "SYS RETAIN"

    def test_kinds(self):
        assert _roundtrip(q.sys.kinds()) == "SYS KINDS"

    def test_edge_kinds(self):
        assert _roundtrip(q.sys.edge_kinds()) == "SYS EDGE KINDS"

    def test_checkpoint(self):
        assert _roundtrip(q.sys.checkpoint()) == "SYS CHECKPOINT"

    def test_rebuild_indices(self):
        assert _roundtrip(q.sys.rebuild_indices()) == "SYS REBUILD INDICES"

    def test_embedders(self):
        assert _roundtrip(q.sys.embedders()) == "SYS EMBEDDERS"

    def test_reembed(self):
        assert _roundtrip(q.sys.reembed()) == "SYS REEMBED"

    def test_snapshots(self):
        assert _roundtrip(q.sys.snapshots()) == "SYS SNAPSHOTS"


class TestSysStats:
    def test_no_target(self):
        assert _roundtrip(q.sys.stats()) == "SYS STATS"

    def test_target_nodes(self):
        assert _roundtrip(q.sys.stats("NODES")) == "SYS STATS NODES"

    def test_target_memory_lowercase(self):
        assert _roundtrip(q.sys.stats("memory")) == "SYS STATS MEMORY"

    def test_invalid_target(self):
        with pytest.raises(ValueError):
            q.sys.stats("FOO")


class TestSysDescribe:
    def test_node(self):
        assert _roundtrip(q.sys.describe("NODE", "memory")) == 'SYS DESCRIBE NODE "memory"'

    def test_edge(self):
        assert _roundtrip(q.sys.describe("EDGE", "mentions")) == 'SYS DESCRIBE EDGE "mentions"'

    def test_invalid_entity(self):
        with pytest.raises(ValueError):
            q.sys.describe("TABLE", "x")


class TestSysQueries:
    def test_slow(self):
        assert _roundtrip(q.sys.slow_queries()) == "SYS SLOW QUERIES"

    def test_slow_with_since_limit(self):
        dsl = _roundtrip(q.sys.slow_queries(since="2024-01-01", limit=10))
        assert 'SINCE "2024-01-01"' in dsl
        assert "LIMIT 10" in dsl

    def test_frequent(self):
        dsl = _roundtrip(q.sys.frequent_queries(limit=5))
        assert dsl == "SYS FREQUENT QUERIES LIMIT 5"

    def test_failed(self):
        assert _roundtrip(q.sys.failed_queries()) == "SYS FAILED QUERIES"


class TestSysExplain:
    def test_basic(self):
        inner = q.remember("x", limit=5)
        dsl = _roundtrip(q.sys.explain(inner))
        assert dsl.startswith("SYS EXPLAIN REMEMBER")

    def test_rejects_write(self):
        with pytest.raises(ValueError):
            q.sys.explain(q.create_node("n", kind="m"))


class TestSysRegister:
    def test_register_node_kind_basic(self):
        dsl = _roundtrip(q.sys.register_node_kind("memory", required={"topic": "string"}))
        assert dsl == 'SYS REGISTER NODE KIND "memory" REQUIRED topic:string'

    def test_register_node_kind_full(self):
        dsl = _roundtrip(q.sys.register_node_kind(
            "memory",
            required={"topic": "string", "importance": "float"},
            optional={"tags": "string"},
            embed="content",
        ))
        assert "OPTIONAL tags:string" in dsl
        assert "EMBED content" in dsl

    def test_register_edge_kind(self):
        dsl = _roundtrip(q.sys.register_edge_kind("mentions", from_kinds=["message"], to_kinds=["entity"]))
        assert dsl == 'SYS REGISTER EDGE KIND "mentions" FROM "message" TO "entity"'

    def test_unregister_node(self):
        assert _roundtrip(q.sys.unregister("NODE", "memory")) == 'SYS UNREGISTER NODE KIND "memory"'


class TestSysClearWal:
    def test_clear_log(self):
        assert _roundtrip(q.sys.clear("LOG")) == "SYS CLEAR LOG"

    def test_clear_cache(self):
        assert _roundtrip(q.sys.clear("CACHE")) == "SYS CLEAR CACHE"

    def test_clear_invalid(self):
        with pytest.raises(ValueError):
            q.sys.clear("EDGES")

    def test_wal_status(self):
        assert _roundtrip(q.sys.wal("STATUS")) == "SYS WAL STATUS"

    def test_wal_replay(self):
        assert _roundtrip(q.sys.wal("REPLAY")) == "SYS WAL REPLAY"


class TestSysExpireContradictions:
    def test_expire(self):
        assert _roundtrip(q.sys.expire()) == "SYS EXPIRE"

    def test_expire_with_where(self):
        dsl = _roundtrip(q.sys.expire(where=F.eq("kind", "working")))
        assert 'WHERE kind = "working"' in dsl

    def test_contradictions(self):
        dsl = _roundtrip(q.sys.contradictions(field="value", group_by="topic"))
        assert dsl == "SYS CONTRADICTIONS FIELD value GROUP BY topic"

    def test_contradictions_with_where(self):
        dsl = _roundtrip(q.sys.contradictions(field="value", group_by="topic", where=F.eq("kind", "belief")))
        assert "WHERE" in dsl


class TestSysSnapshots:
    def test_snapshot(self):
        assert _roundtrip(q.sys.snapshot("before")) == 'SYS SNAPSHOT "before"'

    def test_rollback(self):
        assert _roundtrip(q.sys.rollback_to("before")) == 'SYS ROLLBACK TO "before"'


class TestSysGraphOps:
    def test_duplicates(self):
        assert _roundtrip(q.sys.duplicates()) == "SYS DUPLICATES"

    def test_duplicates_threshold(self):
        dsl = _roundtrip(q.sys.duplicates(threshold=0.95))
        assert "THRESHOLD 0.95" in dsl

    def test_connect(self):
        assert _roundtrip(q.sys.connect()) == "SYS CONNECT"

    def test_connect_with_where_threshold(self):
        dsl = _roundtrip(q.sys.connect(where=F.eq("kind", "memory"), threshold=0.9))
        assert "WHERE" in dsl
        assert "THRESHOLD 0.9" in dsl

    def test_consolidate(self):
        assert _roundtrip(q.sys.consolidate()) == "SYS CONSOLIDATE"

    def test_consolidate_full(self):
        dsl = _roundtrip(q.sys.consolidate(threshold=0.7, min_cluster_size=3))
        assert "THRESHOLD 0.7" in dsl
        assert "MIN_CLUSTER_SIZE 3" in dsl


class TestSysOptimizeEvictLog:
    def test_optimize(self):
        assert _roundtrip(q.sys.optimize()) == "SYS OPTIMIZE"

    def test_optimize_target(self):
        assert _roundtrip(q.sys.optimize("COMPACT")) == "SYS OPTIMIZE COMPACT"

    def test_optimize_invalid(self):
        with pytest.raises(ValueError):
            q.sys.optimize("JUNK")

    def test_evict(self):
        assert _roundtrip(q.sys.evict()) == "SYS EVICT"

    def test_evict_limit(self):
        assert _roundtrip(q.sys.evict(limit=20)) == "SYS EVICT LIMIT 20"

    def test_log(self):
        assert _roundtrip(q.sys.log()) == "SYS LOG"

    def test_log_trace(self):
        dsl = _roundtrip(q.sys.log(trace="abc123", limit=10))
        assert 'TRACE "abc123"' in dsl
        assert "LIMIT 10" in dsl

    def test_log_mutex_filters(self):
        with pytest.raises(ValueError):
            q.sys.log(since="2024-01", trace="abc")


class TestSysCron:
    def test_add(self):
        dsl = _roundtrip(q.sys.cron.add("nightly", schedule="0 3 * * *", query="SYS OPTIMIZE"))
        assert dsl == 'SYS CRON ADD "nightly" SCHEDULE "0 3 * * *" QUERY "SYS OPTIMIZE"'

    def test_delete(self):
        assert _roundtrip(q.sys.cron.delete("nightly")) == 'SYS CRON DELETE "nightly"'

    def test_enable(self):
        assert _roundtrip(q.sys.cron.enable("n")) == 'SYS CRON ENABLE "n"'

    def test_disable(self):
        assert _roundtrip(q.sys.cron.disable("n")) == 'SYS CRON DISABLE "n"'

    def test_list(self):
        assert _roundtrip(q.sys.cron.list()) == "SYS CRON LIST"

    def test_run(self):
        assert _roundtrip(q.sys.cron.run("n")) == 'SYS CRON RUN "n"'


class TestSysEvolve:
    def test_rule_basic(self):
        dsl = _roundtrip(q.sys.evolve.rule(
            "r1",
            when=["recall_hit_rate <= 0.4"],
            then=["RUN SYS REEMBED"],
        ))
        assert 'SYS EVOLVE RULE "r1"' in dsl
        assert "WHEN recall_hit_rate <= 0.4" in dsl
        assert "THEN RUN SYS REEMBED" in dsl

    def test_rule_with_cooldown_priority(self):
        dsl = _roundtrip(q.sys.evolve.rule(
            "r2",
            when=["x > 0.5"],
            then=["SET y = 0.1"],
            cooldown=86400,
            priority=1,
        ))
        assert "COOLDOWN 86400" in dsl
        assert "PRIORITY 1" in dsl

    def test_list(self):
        assert _roundtrip(q.sys.evolve.list()) == "SYS EVOLVE LIST"

    def test_show(self):
        assert _roundtrip(q.sys.evolve.show("r1")) == 'SYS EVOLVE SHOW "r1"'

    def test_enable(self):
        assert _roundtrip(q.sys.evolve.enable("r1")) == 'SYS EVOLVE ENABLE "r1"'

    def test_disable(self):
        assert _roundtrip(q.sys.evolve.disable("r1")) == 'SYS EVOLVE DISABLE "r1"'

    def test_delete(self):
        assert _roundtrip(q.sys.evolve.delete("r1")) == 'SYS EVOLVE DELETE "r1"'

    def test_history(self):
        assert _roundtrip(q.sys.evolve.history()) == "SYS EVOLVE HISTORY"

    def test_history_limit(self):
        assert _roundtrip(q.sys.evolve.history(limit=10)) == "SYS EVOLVE HISTORY LIMIT 10"

    def test_reset(self):
        assert _roundtrip(q.sys.evolve.reset()) == "SYS EVOLVE RESET"


class TestVault:
    def test_new(self):
        assert _roundtrip(q.vault.new("Doc")) == 'VAULT NEW "Doc"'

    def test_new_with_kind_tags(self):
        dsl = _roundtrip(q.vault.new("Doc", kind="context", tags="projectX,q3"))
        assert 'KIND "context"' in dsl
        assert 'TAGS "projectX,q3"' in dsl

    def test_read(self):
        assert _roundtrip(q.vault.read("Doc")) == 'VAULT READ "Doc"'

    def test_write(self):
        dsl = _roundtrip(q.vault.write("Doc", section="Summary", content="text"))
        assert dsl == 'VAULT WRITE "Doc" SECTION "Summary" CONTENT "text"'

    def test_append(self):
        dsl = _roundtrip(q.vault.append("Doc", section="Log", content="more"))
        assert dsl == 'VAULT APPEND "Doc" SECTION "Log" CONTENT "more"'

    def test_search(self):
        dsl = _roundtrip(q.vault.search("deploy", limit=5))
        assert dsl == 'VAULT SEARCH "deploy" LIMIT 5'

    def test_backlinks(self):
        assert _roundtrip(q.vault.backlinks("Doc")) == 'VAULT BACKLINKS "Doc"'

    def test_list(self):
        assert _roundtrip(q.vault.list()) == "VAULT LIST"

    def test_list_where_order_limit(self):
        dsl = _roundtrip(q.vault.list(where=F.eq("kind", "note"), order_by="title", limit=5))
        assert "WHERE" in dsl
        assert "ORDER BY title" in dsl
        assert "LIMIT 5" in dsl

    def test_sync(self):
        assert _roundtrip(q.vault.sync()) == "VAULT SYNC"

    def test_daily(self):
        assert _roundtrip(q.vault.daily()) == "VAULT DAILY"

    def test_archive(self):
        assert _roundtrip(q.vault.archive("Doc")) == 'VAULT ARCHIVE "Doc"'
