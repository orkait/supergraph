"""DSL executor: auto-dispatch via handler registry.

Combines all domain handler mixins via multiple inheritance.
New commands self-register via @handles decorator - no manual dispatch dict.
"""

from supergraph.dsl.ast_nodes import (
    VaultNew, VaultRead, VaultWrite, VaultAppend,
    VaultSearch, VaultBacklinks, VaultList,
    VaultSync, VaultDaily, VaultArchive,
)
from supergraph.core.errors import SuperGraphError
from supergraph.core.types import Result
from supergraph.dsl.executor_base import ExecutorBase

from supergraph.dsl.handlers import (
    DISPATCH,
    NodeHandlers,
    EdgeHandlers,
    TraversalHandlers,
    PatternHandlers,
    AggregationHandlers,
    IntelligenceHandlers,
    BeliefHandlers,
    MutationHandlers,
    ContextHandlers,
    IngestHandlers,
)


_VAULT_TYPES = (VaultNew, VaultRead, VaultWrite, VaultAppend,
                VaultSearch, VaultBacklinks, VaultList,
                VaultSync, VaultDaily, VaultArchive)


class Executor(
    NodeHandlers,
    EdgeHandlers,
    TraversalHandlers,
    PatternHandlers,
    AggregationHandlers,
    IntelligenceHandlers,
    BeliefHandlers,
    MutationHandlers,
    ContextHandlers,
    IngestHandlers,
    ExecutorBase,
):
    """Full executor combining all domain handlers via auto-dispatch registry."""

    _vault_executor = None

    def _dispatch(self, ast) -> Result:
        if isinstance(ast, _VAULT_TYPES):
            if not self._vault_executor:
                raise SuperGraphError("Vault not configured. Use SuperGraph(vault='./notes')")
            return self._vault_executor.dispatch(ast)

        t = type(ast)
        handler = DISPATCH.get(t)
        if handler is None:
            for base in t.__mro__[1:]:
                handler = DISPATCH.get(base)
                if handler is not None:
                    break
        if handler is None:
            raise SuperGraphError(f"Unknown AST node type: {t.__name__}")
        return handler(self, ast)
