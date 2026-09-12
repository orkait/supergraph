"""Metacognitive evolution layer.

Moved from ``supergraph.evolve`` into ``supergraph.core.evolve`` so the
self-tuning runtime lives next to the subsystems it tunes. Public surface
unchanged: every previously-importable symbol is re-exported here and
through ``supergraph.evolve`` for backwards compat.
"""

from supergraph.core.evolve._impl import (
    Action,
    Condition,
    EvolutionEngine,
    EvolutionRule,
    KNOWN_SIGNALS,
    TUNABLE_PARAMS,
)
from supergraph.core.evolve._defaults import STARTER_RULES

__all__ = [
    "Action",
    "Condition",
    "EvolutionEngine",
    "EvolutionRule",
    "KNOWN_SIGNALS",
    "TUNABLE_PARAMS",
    "STARTER_RULES",
]
