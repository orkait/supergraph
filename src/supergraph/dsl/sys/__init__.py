
from supergraph.dsl.sys._registry import SYS_DISPATCH, handles_sys

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
