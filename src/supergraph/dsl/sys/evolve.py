
from __future__ import annotations

import logging


from supergraph.core.errors import SuperGraphError
from supergraph.core.types import Result

logger = logging.getLogger(__name__)

from supergraph.dsl.ast_nodes import (
    SysEvolveDelete,
    SysEvolveDisable,
    SysEvolveEnable,
    SysEvolveHistory,
    SysEvolveList,
    SysEvolveReset,
    SysEvolveRule,
    SysEvolveShow,
)
from supergraph.dsl.sys._registry import handles_sys


class SysEvolveHandlers:
    def _get_engine(self):
        if self._evolution_engine is None:
            raise SuperGraphError("Evolution engine not initialized")
        return self._evolution_engine

    @handles_sys(SysEvolveRule)
    def _evolve_rule(self, q: SysEvolveRule) -> Result:
        from supergraph.core.evolve import EvolutionRule, Condition, Action
        engine = self._get_engine()
        rule = EvolutionRule(
            name=q.name,
            cooldown=q.cooldown,
            priority=q.priority,
        )
        for c in q.conditions:
            rule.conditions.append(Condition(
                signal=c["signal"],
                operator=c["operator"],
                value=c["value"],
            ))
        for a in q.actions:
            rule.actions.append(Action(
                kind=a["kind"],
                param=a["param"],
                value=a.get("value"),
                delta=a.get("delta", 0.0),
                until=a.get("until"),
            ))
        err = engine.add_rule(rule)
        if err:
            return Result(kind="error", data=err, count=0)
        return Result(kind="ok", data={"created": q.name}, count=1)

    @handles_sys(SysEvolveList)
    def _evolve_list(self, q: SysEvolveList) -> Result:
        engine = self._get_engine()
        rules = engine.list_rules()
        return Result(kind="rules", data=rules, count=len(rules))

    @handles_sys(SysEvolveShow)
    def _evolve_show(self, q: SysEvolveShow) -> Result:
        engine = self._get_engine()
        rule = engine.get_rule(q.name)
        if rule is None:
            return Result(kind="error", data=f"rule not found: '{q.name}'", count=0)
        return Result(kind="rule", data=rule, count=1)

    @handles_sys(SysEvolveEnable)
    def _evolve_enable(self, q: SysEvolveEnable) -> Result:
        engine = self._get_engine()
        if not engine.enable_rule(q.name):
            return Result(kind="error", data=f"rule not found: '{q.name}'", count=0)
        return Result(kind="ok", data={"enabled": q.name}, count=1)

    @handles_sys(SysEvolveDisable)
    def _evolve_disable(self, q: SysEvolveDisable) -> Result:
        engine = self._get_engine()
        if not engine.disable_rule(q.name):
            return Result(kind="error", data=f"rule not found: '{q.name}'", count=0)
        return Result(kind="ok", data={"disabled": q.name}, count=1)

    @handles_sys(SysEvolveDelete)
    def _evolve_delete(self, q: SysEvolveDelete) -> Result:
        engine = self._get_engine()
        if not engine.delete_rule(q.name):
            return Result(kind="error", data=f"rule not found: '{q.name}'", count=0)
        return Result(kind="ok", data={"deleted": q.name}, count=1)

    @handles_sys(SysEvolveHistory)
    def _evolve_history(self, q: SysEvolveHistory) -> Result:
        engine = self._get_engine()
        entries = engine.history(limit=q.limit)
        return Result(kind="history", data=entries, count=len(entries))

    @handles_sys(SysEvolveReset)
    def _evolve_reset(self, q: SysEvolveReset) -> Result:
        engine = self._get_engine()
        engine.reset()
        return Result(kind="ok", data={"reset": True}, count=0)
