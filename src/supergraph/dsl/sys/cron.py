
from __future__ import annotations

import logging


from supergraph.core.errors import SuperGraphError
from supergraph.core.types import Result

logger = logging.getLogger(__name__)

from supergraph.dsl.ast_nodes import (
    SysCronAdd,
    SysCronDelete,
    SysCronDisable,
    SysCronEnable,
    SysCronList,
    SysCronRun,
)
from supergraph.dsl.sys._registry import handles_sys


class SysCronHandlers:
    @handles_sys(SysCronAdd)
    def _cron_add(self, q: SysCronAdd) -> Result:
        if not self._cron:
            raise SuperGraphError("CRON not configured. Use SuperGraph(queued=True)")
        data = self._cron.add(q.name, q.schedule, q.query)
        return Result(kind="ok", data=data, count=1)

    @handles_sys(SysCronDelete)
    def _cron_delete(self, q: SysCronDelete) -> Result:
        if not self._cron:
            raise SuperGraphError("CRON not configured")
        self._cron.delete(q.name)
        return Result(kind="ok", data={"deleted": q.name}, count=1)

    @handles_sys(SysCronEnable)
    def _cron_enable(self, q: SysCronEnable) -> Result:
        if not self._cron:
            raise SuperGraphError("CRON not configured")
        self._cron.enable(q.name)
        return Result(kind="ok", data={"enabled": q.name}, count=1)

    @handles_sys(SysCronDisable)
    def _cron_disable(self, q: SysCronDisable) -> Result:
        if not self._cron:
            raise SuperGraphError("CRON not configured")
        self._cron.disable(q.name)
        return Result(kind="ok", data={"disabled": q.name}, count=1)

    @handles_sys(SysCronList)
    def _cron_list(self, q: SysCronList) -> Result:
        if not self._cron:
            raise SuperGraphError("CRON not configured")
        jobs = self._cron.list_jobs()
        return Result(kind="cron_jobs", data=jobs, count=len(jobs))

    @handles_sys(SysCronRun)
    def _cron_run(self, q: SysCronRun) -> Result:
        if not self._cron:
            raise SuperGraphError("CRON not configured")
        self._cron.run_now(q.name)
        return Result(kind="ok", data={"triggered": q.name}, count=1)

