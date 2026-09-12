
from __future__ import annotations

import logging


from supergraph.core.types import Result

logger = logging.getLogger(__name__)

from supergraph.dsl.ast_nodes import (
    SysRegisterEdgeKind,
    SysRegisterNodeKind,
    SysUnregister,
)
from supergraph.dsl.sys._registry import handles_sys


class SysSchemaHandlers:
    @handles_sys(SysRegisterNodeKind)
    def _register_node_kind(self, q: SysRegisterNodeKind) -> Result:
        self.schema.register_node_kind(q.kind, q.required, q.optional,
                                       embed_field=q.embed_field)
        type_map = {"string": "int32_interned", "int": "int64", "float": "float64"}
        for item in q.required + q.optional:
            if isinstance(item, tuple):
                name, type_name = item
            else:
                name, type_name = item, None
            if type_name and type_name in type_map:
                self.store.columns.declare_column(name, type_map[type_name])
        return Result(kind="ok", data=None, count=0)

    @handles_sys(SysRegisterEdgeKind)
    def _register_edge_kind(self, q: SysRegisterEdgeKind) -> Result:
        self.schema.register_edge_kind(q.kind, q.from_kinds, q.to_kinds)
        return Result(kind="ok", data=None, count=0)

    @handles_sys(SysUnregister)
    def _unregister(self, q: SysUnregister) -> Result:
        if q.entity_type == "NODE":
            self.schema.unregister_node_kind(q.kind)
        else:
            self.schema.unregister_edge_kind(q.kind)
        return Result(kind="ok", data=None, count=0)
