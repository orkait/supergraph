import pytest
from supergraph import SuperGraph
from supergraph.cron import CronScheduler

pytestmark = pytest.mark.needs_scheduler


class TestCronCRUD:
    def test_add_and_list(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), queued=True)
        gs.execute('SYS CRON ADD "expire" SCHEDULE "0 * * * *" QUERY "SYS EXPIRE"')
        result = gs.execute('SYS CRON LIST')
        assert result.kind == "cron_jobs"
        assert len(result.data) == 1
        job = result.data[0]
        assert job["name"] == "expire"
        assert job["schedule"] == "0 * * * *"
        assert job["query"] == "SYS EXPIRE"
        assert job["enabled"] is True
        gs.close()

    def test_delete_job(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), queued=True)
        gs.execute('SYS CRON ADD "test" SCHEDULE "@hourly" QUERY "SYS STATS"')
        gs.execute('SYS CRON DELETE "test"')
        result = gs.execute('SYS CRON LIST')
        assert len(result.data) == 0
        gs.close()

    def test_enable_disable(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), queued=True)
        gs.execute('SYS CRON ADD "j" SCHEDULE "*/5 * * * *" QUERY "SYS STATS"')
        gs.execute('SYS CRON DISABLE "j"')
        result = gs.execute('SYS CRON LIST')
        assert result.data[0]["enabled"] is False
        gs.execute('SYS CRON ENABLE "j"')
        result = gs.execute('SYS CRON LIST')
        assert result.data[0]["enabled"] is True
        gs.close()

    def test_duplicate_name_rejected(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), queued=True)
        gs.execute('SYS CRON ADD "dup" SCHEDULE "@daily" QUERY "SYS STATS"')
        with pytest.raises(Exception) as exc_info:
            gs.execute('SYS CRON ADD "dup" SCHEDULE "@hourly" QUERY "SYS EXPIRE"')
        msg = str(exc_info.value).lower()
        assert "already exists" in msg or "duplicate" in msg or "unique" in msg.upper()
        gs.close()

    def test_invalid_cron_expression_rejected(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), queued=True)
        with pytest.raises(Exception) as exc_info:
            gs.execute('SYS CRON ADD "bad" SCHEDULE "not-a-cron" QUERY "SYS STATS"')
        msg = str(exc_info.value).lower()
        assert "invalid" in msg or "cron" in msg
        gs.close()

    def test_run_now(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), queued=True)
        gs.execute('CREATE NODE "x" kind = "test"')
        gs.execute('SYS CRON ADD "stats" SCHEDULE "0 0 1 1 *" QUERY "SYS STATS"')
        result = gs.execute('SYS CRON RUN "stats"')
        assert result.kind == "ok"
        gs.close()


class TestCronExpressions:

    def test_standard_five_field(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), queued=True)
        gs.execute('SYS CRON ADD "a" SCHEDULE "30 2 * * 1-5" QUERY "SYS STATS"')
        result = gs.execute('SYS CRON LIST')
        assert result.data[0]["schedule"] == "30 2 * * 1-5"
        gs.close()

    def test_at_shortcuts(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), queued=True)
        for name, sched in [("h", "@hourly"), ("d", "@daily"), ("w", "@weekly"), ("m", "@monthly"), ("y", "@yearly")]:
            gs.execute(f'SYS CRON ADD "{name}" SCHEDULE "{sched}" QUERY "SYS STATS"')
        result = gs.execute('SYS CRON LIST')
        assert len(result.data) == 5
        gs.close()

    def test_step_and_range(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), queued=True)
        gs.execute('SYS CRON ADD "s" SCHEDULE "*/15 9-17 * * MON-FRI" QUERY "SYS STATS"')
        result = gs.execute('SYS CRON LIST')
        assert result.data[0]["schedule"] == "*/15 9-17 * * MON-FRI"
        gs.close()


class TestCronPersistence:
    def test_jobs_survive_restart(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), queued=True)
        gs.execute('SYS CRON ADD "persist" SCHEDULE "@daily" QUERY "SYS EXPIRE"')
        gs.close()

        gs2 = SuperGraph(path=str(tmp_path / "db"), queued=True)
        result = gs2.execute('SYS CRON LIST')
        assert len(result.data) == 1
        assert result.data[0]["name"] == "persist"
        gs2.close()


class TestCronSchedulerUnit:
    def test_tick_fires_due_job(self, tmp_path):
        from concurrent.futures import Future
        from supergraph.persistence.database import open_database

        results = []

        def fake_submit(query):
            f = Future()
            results.append(query)
            f.set_result(None)
            return f

        conn = open_database(str(tmp_path / "test.db"))
        sched = CronScheduler(conn, fake_submit)
        sched.add("test_job", "* * * * *", "SYS STATS")
        conn.execute("UPDATE cron_jobs SET next_run = 0 WHERE name = 'test_job'")
        conn.commit()
        sched._tick()
        assert "SYS STATS" in results
        conn.close()


class TestCronRequiresQueued:
    def test_cron_not_available_without_queued(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), queued=False)
        with pytest.raises(Exception) as exc_info:
            gs.execute('SYS CRON LIST')
        msg = str(exc_info.value).lower()
        assert "cron" in msg or "queued" in msg
        gs.close()
