"""SYS command handlers, sharded by domain.

executor_system.py used to be a 1267-LOC class with ~45 handler methods
on one body. Now SystemExecutor composes six mixin modules here, each
self-registering its handlers via @handles_sys(AstType).
"""

from supergraph.dsl.sys._registry import SYS_DISPATCH, handles_sys

# Importing these modules registers every @handles_sys decorator in them.
from supergraph.dsl.sys.queries import SysQueryHandlers
from supergraph.dsl.sys.schema import SysSchemaHandlers
from supergraph.dsl.sys.lifecycle import SysLifecycleHandlers
from supergraph.dsl.sys.pipeline import SysPipelineHandlers
from supergraph.dsl.sys.cron import SysCronHandlers
from supergraph.dsl.sys.evolve import SysEvolveHandlers

__all__ = [
    "SYS_DISPATCH",
    "handles_sys",
    "SysQueryHandlers",
    "SysSchemaHandlers",
    "SysLifecycleHandlers",
    "SysPipelineHandlers",
    "SysCronHandlers",
    "SysEvolveHandlers",
]
