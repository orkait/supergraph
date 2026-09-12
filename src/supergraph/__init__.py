
__version__ = "0.7.0"

from .core import compute_profile as _compute_profile_init  # noqa: F401

from . import gpu  # noqa: F401  - expose supergraph.gpu.{setup, is_ready, status}
from . import pro  # noqa: F401  - expose supergraph.pro.{ProSpec, resolve, ...}
from .store import SuperGraph
from .research import Research
from .core.store import CoreStore
from .core.schema import SchemaRegistry
from .core.types import Result, Edge
from .dsl.parser import parse, clear_cache
from .dsl.executor import Executor
from .dsl.executor_system import SystemExecutor
from .core.errors import (
    SuperGraphError, QueryError, NodeNotFound, NodeExists,
    CeilingExceeded, VersionMismatch, SchemaError,
    CostThresholdExceeded, BatchRollback, AggregationError,
    VectorError, EmbedderRequired, VectorNotFound,
    OptimizationInProgress, StoreInUse,
)
from .core.memory import DEFAULT_CEILING_BYTES
from .config import (
    SuperGraphConfig, load_config, save_config,
    CoreConfig, VectorConfig, DocumentConfig, DslConfig,
    VaultConfig, PersistenceConfig, RetentionConfig, ServerConfig,
)
from .query import (
    q, F, Query, Time, TimeExpr,
    P, Pattern, agg, AggFunc, HavingExpr,
    EvolveWhen, EvolveThen, EvolveCondition, EvolveAction,
    register_verb,
)

__all__ = [
    "SuperGraph", "Research", "CoreStore", "SchemaRegistry",
    "Result", "Edge",
    "parse", "clear_cache", "Executor", "SystemExecutor",
    "SuperGraphError", "QueryError", "NodeNotFound", "NodeExists",
    "CeilingExceeded", "VersionMismatch", "SchemaError",
    "CostThresholdExceeded", "BatchRollback", "AggregationError",
    "VectorError", "EmbedderRequired", "VectorNotFound",
    "OptimizationInProgress", "StoreInUse",
    "DEFAULT_CEILING_BYTES",
    "SuperGraphConfig", "load_config", "save_config",
    "CoreConfig", "VectorConfig", "DocumentConfig", "DslConfig",
    "VaultConfig", "PersistenceConfig", "RetentionConfig", "ServerConfig",
    "q", "F", "Query", "Time", "TimeExpr",
    "P", "Pattern", "agg", "AggFunc", "HavingExpr",
    "EvolveWhen", "EvolveThen", "EvolveCondition", "EvolveAction",
    "register_verb",
]
