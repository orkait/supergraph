"""Handler registry and domain-specific handler mixins.

Importing this package triggers @handles registration for all handlers.
"""

from supergraph.dsl.handlers._registry import DISPATCH, WRITE_OPS, is_write_op
from supergraph.dsl.handlers.nodes import NodeHandlers
from supergraph.dsl.handlers.edges import EdgeHandlers
from supergraph.dsl.handlers.traversal import TraversalHandlers
from supergraph.dsl.handlers.pattern import PatternHandlers
from supergraph.dsl.handlers.aggregation import AggregationHandlers
from supergraph.dsl.handlers.intelligence import IntelligenceHandlers
from supergraph.dsl.handlers.beliefs import BeliefHandlers
from supergraph.dsl.handlers.mutations import MutationHandlers
from supergraph.dsl.handlers.context import ContextHandlers
from supergraph.dsl.handlers.ingest import IngestHandlers

__all__ = [
    "DISPATCH", "WRITE_OPS", "is_write_op",
    "NodeHandlers", "EdgeHandlers", "TraversalHandlers",
    "PatternHandlers", "AggregationHandlers", "IntelligenceHandlers",
    "BeliefHandlers", "MutationHandlers", "ContextHandlers", "IngestHandlers",
]
