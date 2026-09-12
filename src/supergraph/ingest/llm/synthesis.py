"""Stable internal API over BonsaiIngestor's @-verb parse + DSL synthesis.

Cloud ingestion reuses the EXACT same parse / synthesize / postproc logic as
the local Bonsai path so the resulting graph shape is identical regardless of
backend. These are re-exported (not reimplemented) from bonsai_ingestor;
importing that module is cheap because the heavy llama-cpp import is deferred
to BonsaiIngestor._ensure_llm and is never triggered here.

Coupling note: these are currently underscore-prefixed in bonsai_ingestor.
This shim is the single place that depends on those names, so a future
extraction of a shared synthesis module only has to update this file.
"""
from supergraph.bonsai_ingestor import (  # noqa: F401
    ParsedTurn,
    FactState,
    IngestResult,
    BonsaiError,
    IngestEmpty,
    IngestOverflow,
    _parse_verb_output as parse_verb_output,
    _synthesize_dsl as synthesize_dsl,
    _render_known_facts_block as render_known_facts_block,
    _scrape_belief_updates as scrape_belief_updates,
    _strip_think as strip_think,
    _UPSERT_RE,
    _ASSERT_RE,
    _RETRACT_RE,
    _ENT_FROM_ID_RE,
)

# Canonical error name going forward; BonsaiError stays as the alias so
# existing `except BonsaiError` callers keep working.
IngestError = BonsaiError
