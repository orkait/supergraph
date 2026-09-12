
import numpy as np

from supergraph.dsl.handlers._registry import handles
from supergraph.dsl.ast_nodes import (
    BindContext, DiscardContext, BindNamespace, DiscardNamespace,
)
from supergraph.core.types import Result
from supergraph.core.errors import NodeNotFound


class ContextHandlers:

    @handles(BindContext, write=True)
    def _bind_context(self, q: BindContext) -> Result:
        if getattr(self.store, "_active_namespace", None) is not None:
            from supergraph.core.errors import SuperGraphError
            raise SuperGraphError(
                "cannot BIND CONTEXT while a NAMESPACE is bound (filters would "
                "AND to empty); DISCARD NAMESPACE first"
            )
        self.store._active_context = q.name
        return Result(kind="ok", data={"context": q.name}, count=0)

    @handles(BindNamespace, write=True)
    def _bind_namespace(self, q: BindNamespace) -> Result:
        if self.store._active_context is not None:
            from supergraph.core.errors import SuperGraphError
            raise SuperGraphError(
                "cannot BIND NAMESPACE while a CONTEXT is bound (filters would "
                "AND to empty); DISCARD CONTEXT first"
            )
        self.store._active_namespace = q.name
        return Result(kind="ok", data={"namespace": q.name}, count=0)

    @handles(DiscardNamespace, write=True)
    def _discard_namespace(self, q: DiscardNamespace) -> Result:
        self.store._active_namespace = None
        return Result(kind="ok", data={"namespace_unbound": q.name}, count=0)

    @handles(DiscardContext, write=True)
    def _discard_context(self, q: DiscardContext) -> Result:
        deleted_count = 0
        n = self.store._next_slot
        if n > 0 and self.store.columns.has_column("__context__"):
            ctx_col = self.store.columns.get_column("__context__", n)
            if ctx_col is not None:
                col_data, col_pres, _ = ctx_col
                if q.name not in self.store.string_table:
                    self.store._active_context = None
                    return Result(
                        kind="ok",
                        data={"discarded": q.name, "deleted": 0},
                        count=0,
                    )
                ctx_id = self.store.string_table.intern(q.name)
                ctx_mask = col_pres & (col_data == ctx_id)
                slots_to_delete = np.nonzero(ctx_mask)[0]
                for slot in slots_to_delete:
                    nid = self.store._slot_to_id(int(slot))
                    if nid:
                        try:
                            self.store.delete_node(nid)
                            deleted_count += 1
                        except NodeNotFound:
                            pass

        self.store._active_context = None
        return Result(kind="ok", data={"discarded": q.name, "deleted": deleted_count}, count=deleted_count)
