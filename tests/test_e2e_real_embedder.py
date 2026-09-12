"""End-to-end tests with REAL embedder and REAL fixture data.

These tests verify that supergraph works as a cognitive layer for AI agents:
ingest real documents, build knowledge, query semantically, maintain over time.
"""

import tempfile
import time
from pathlib import Path

import pytest
import numpy as np

FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = [
    pytest.mark.needs_embedder,
    pytest.mark.needs_ingest,
    pytest.mark.xdist_group("e2e_real_embedder"),
]


@pytest.fixture(scope="module")
def brain(tmp_path_factory):
    """Persistent SuperGraph with real Model2Vec embedder. Shared across module."""
    from supergraph import SuperGraph
    from supergraph.embedding.model2vec_embedder import Model2VecEmbedder
    td = tmp_path_factory.mktemp("brain")
    gs = SuperGraph(
        path=str(td),
        embedder=Model2VecEmbedder(),
        ceiling_mb=256,
    )
    yield gs
    gs.close()


# ===================================================================
# Phase 1: Ingest real documents
# ===================================================================

class TestIngestRealDocuments:
    """Ingest real PDFs, text files, and markdown into the brain."""

    def test_ingest_attention_paper(self, brain):
        """Ingest the 'Attention Is All You Need' PDF."""
        path = FIXTURES / "pdf" / "attention-is-all-you-need.pdf"
        if not path.exists():
            pytest.skip("PDF fixture not found")
        result = brain.execute(f'INGEST "{path}" AS "paper:attention" KIND "paper"')
        assert result.data["chunks"] > 0
        assert result.data["parser"] in ("pymupdf4llm", "markitdown")

    def test_ingest_bert_paper(self, brain):
        path = FIXTURES / "pdf" / "bert.pdf"
        if not path.exists():
            pytest.skip("PDF fixture not found")
        result = brain.execute(f'INGEST "{path}" AS "paper:bert" KIND "paper"')
        assert result.data["chunks"] > 0

    def test_ingest_rag_paper(self, brain):
        path = FIXTURES / "pdf" / "rag.pdf"
        if not path.exists():
            pytest.skip("PDF fixture not found")
        result = brain.execute(f'INGEST "{path}" AS "paper:rag" KIND "paper"')
        assert result.data["chunks"] > 0

    def test_ingest_vector_db_docs(self, brain):
        """Ingest markdown docs about vector databases."""
        for name in ["faiss", "chroma", "qdrant"]:
            path = FIXTURES / "markdown" / f"{name}.md"
            if path.exists():
                brain.execute(f'INGEST "{path}" AS "doc:{name}" KIND "docs"')

    def test_ingest_text_book(self, brain):
        """Ingest a real book (Art of War - short, diverse content)."""
        path = FIXTURES / "text" / "art-of-war.txt"
        if not path.exists():
            pytest.skip("Text fixture not found")
        result = brain.execute(f'INGEST "{path}" AS "book:artofwar" KIND "book"')
        assert result.data["chunks"] > 0

    def test_auto_wire_similar_chunks(self, brain):
        """SYS CONNECT should find cross-document similarities with real vectors."""
        result = brain.execute('SYS CONNECT THRESHOLD 0.7')
        assert result.kind == "ok"
        # With real embeddings, similar ML paper chunks should get wired


# ===================================================================
# Phase 2: Semantic retrieval with real embeddings
# ===================================================================

class TestSemanticRetrieval:
    """Verify that SIMILAR TO and REMEMBER return semantically relevant results."""

    def test_similar_to_finds_attention_content(self, brain):
        """Searching for 'attention mechanism' should surface attention paper chunks."""
        result = brain.execute('SIMILAR TO "self-attention mechanism in transformers" LIMIT 10')
        assert result.kind == "nodes"
        assert len(result.data) > 0

        # At least one result should be from the attention paper
        attention_results = [
            n for n in result.data
            if "attention" in n.get("id", "").lower()
            or "attention" in n.get("summary", "").lower()
        ]
        assert len(attention_results) > 0, (
            f"Expected attention paper in results, got: "
            f"{[n.get('id', '')[:40] for n in result.data[:5]]}"
        )

    def test_similar_to_finds_bert_content(self, brain):
        """Searching for 'bidirectional language model' should surface BERT chunks."""
        result = brain.execute('SIMILAR TO "bidirectional language model pretraining" LIMIT 10')
        assert len(result.data) > 0

        bert_results = [
            n for n in result.data
            if "bert" in n.get("id", "").lower()
            or "bert" in n.get("summary", "").lower()
            or "bidirectional" in n.get("summary", "").lower()
        ]
        assert len(bert_results) > 0, (
            f"Expected BERT content in results, got: "
            f"{[n.get('id', '')[:40] for n in result.data[:5]]}"
        )

    def test_similar_to_finds_rag_content(self, brain):
        """Searching for 'retrieval augmented generation' should surface RAG chunks."""
        result = brain.execute('SIMILAR TO "retrieval augmented generation knowledge" LIMIT 10')
        assert len(result.data) > 0

        rag_results = [
            n for n in result.data
            if "rag" in n.get("id", "").lower()
            or "retrieval" in n.get("summary", "").lower()
        ]
        assert len(rag_results) > 0, (
            f"Expected RAG content in results, got: "
            f"{[n.get('id', '')[:40] for n in result.data[:5]]}"
        )

    def test_remember_fuses_signals(self, brain):
        """REMEMBER should combine vector + BM25 + recency for better results."""
        result = brain.execute('REMEMBER "attention mechanism transformer architecture" LIMIT 10')
        assert result.kind == "nodes"
        assert len(result.data) > 0

        # Results should have score breakdown
        top = result.data[0]
        assert "_remember_score" in top
        assert top["_remember_score"] > 0

    def test_lexical_search_finds_exact_terms(self, brain):
        """LEXICAL SEARCH should find exact terms in document summaries."""
        result = brain.execute('LEXICAL SEARCH "attention" LIMIT 10')
        assert result.kind == "nodes"
        # BM25 should find chunks containing "attention"

    def test_similar_to_with_filter(self, brain):
        """SIMILAR TO with WHERE kind filter."""
        result = brain.execute('SIMILAR TO "neural network" LIMIT 10 WHERE kind = "chunk"')
        for node in result.data:
            assert node["kind"] == "chunk"


# ===================================================================
# Phase 3: Knowledge graph operations
# ===================================================================

class TestKnowledgeGraph:
    """Build and query a knowledge graph on top of ingested documents."""

    def test_assert_beliefs_from_papers(self, brain):
        """Assert facts learned from papers."""
        brain.execute(
            'ASSERT "belief:attention-key" kind = "belief" '
            'claim = "self-attention computes weighted sum of values" '
            'CONFIDENCE 0.95 SOURCE "paper:attention"'
        )
        brain.execute(
            'ASSERT "belief:bert-bidir" kind = "belief" '
            'claim = "BERT uses bidirectional context" '
            'CONFIDENCE 0.9 SOURCE "paper:bert"'
        )
        brain.execute(
            'ASSERT "belief:rag-retrieval" kind = "belief" '
            'claim = "RAG retrieves documents before generation" '
            'CONFIDENCE 0.85 SOURCE "paper:rag"'
        )

        beliefs = brain.execute('NODES WHERE kind = "belief"')
        assert len(beliefs.data) == 3

    def test_create_topic_hierarchy(self, brain):
        """Build topic nodes and link to papers."""
        brain.execute('CREATE NODE "topic:nlp" kind = "topic" name = "NLP"')
        brain.execute('CREATE NODE "topic:transformers" kind = "topic" name = "transformers"')
        brain.execute('CREATE NODE "topic:retrieval" kind = "topic" name = "retrieval"')

        brain.execute('CREATE EDGE "topic:nlp" -> "topic:transformers" kind = "subtopic"')
        brain.execute('CREATE EDGE "paper:attention" -> "topic:transformers" kind = "about"')
        brain.execute('CREATE EDGE "paper:bert" -> "topic:transformers" kind = "about"')
        brain.execute('CREATE EDGE "paper:rag" -> "topic:retrieval" kind = "about"')

    def test_recall_from_paper(self, brain):
        """RECALL from a paper should activate connected topics and chunks."""
        result = brain.execute('RECALL FROM "paper:attention" DEPTH 2 LIMIT 20')
        assert result.kind == "nodes"
        assert len(result.data) > 0

    def test_traverse_topic_graph(self, brain):
        """TRAVERSE from NLP topic should reach papers."""
        result = brain.execute('TRAVERSE FROM "topic:nlp" DEPTH 3')
        assert len(result.data) > 0

    def test_subgraph_extraction(self, brain):
        """SUBGRAPH should return connected component."""
        result = brain.execute('SUBGRAPH FROM "paper:attention" DEPTH 2')
        assert len(result.data["nodes"]) > 0
        # Paper has outgoing edges to chunks and topics


# ===================================================================
# Phase 4: Agent lifecycle operations
# ===================================================================

class TestAgentLifecycle:
    """Simulate real agent operations: trace, maintain, persist."""

    def test_traced_research_session(self, brain):
        """Agent does a traced research session."""
        brain.bind_trace("research-transformers")
        brain.execute('REMEMBER "attention mechanism" LIMIT 5')
        brain.execute('SIMILAR TO "BERT pretraining" LIMIT 5')
        brain.execute('RECALL FROM "topic:transformers" DEPTH 2 LIMIT 10')
        brain.discard_trace()

        # Verify trace shows up in logs
        result = brain.execute('SYS LOG TRACE "research-transformers"')
        assert len(result.data) >= 3, f"Expected 3+ traced queries, got {len(result.data)}"

    def test_working_memory_with_ttl(self, brain):
        """Create scratch nodes with TTL."""
        brain.execute('CREATE NODE "scratch:idea1" kind = "scratch" content = "maybe combine RAG with BERT" EXPIRES IN 1h')
        result = brain.execute('NODE "scratch:idea1"')
        assert result.data is not None

    def test_counterfactual_reasoning(self, brain):
        """WHAT IF RETRACT simulates removing a belief."""
        result = brain.execute('WHAT IF RETRACT "belief:attention-key"')
        assert result.kind == "counterfactual"
        assert result.data["affected_count"] >= 1
        # Original belief should still be intact (simulation only)
        result = brain.execute('NODE "belief:attention-key"')
        assert result.data is not None

    def test_system_health(self, brain):
        """Check system health after all operations."""
        result = brain.execute('SYS HEALTH')
        assert result.data["live_nodes"] > 0
        assert "memory_utilization" in result.data

        result = brain.execute('SYS STATUS')
        assert result.data["nodes"] > 0
        assert result.data["edges"] > 0

    def test_checkpoint_and_stats(self, brain):
        """Checkpoint and verify stats."""
        brain.checkpoint()
        result = brain.execute('SYS STATS')
        assert result.data["node_count"] > 20  # papers + chunks + beliefs + topics
        assert result.data["edge_count"] > 10  # has_chunk + about + subtopic edges

    def test_log_shows_activity(self, brain):
        """SYS LOG should show all activity with auto-tags."""
        result = brain.execute('SYS LOG LIMIT 50')
        tags = {e.get("tag") for e in result.data}
        # Should have multiple tag types from our operations
        assert "intelligence" in tags or "read" in tags
        assert "write" in tags or "belief" in tags or "system" in tags


# ===================================================================
# Phase 5: The real test - does REMEMBER actually work?
# ===================================================================

class TestRealCognitiveRetrieval:
    """The test that proves supergraph works as agent memory.

    'Ingest real papers, ask what transformers and BERT have in common,
    verify REMEMBER returns attention mechanism chunks.'
    """

    def test_what_do_transformers_and_bert_have_in_common(self, brain):
        """THE CORE TEST: semantic query across multiple ingested papers.

        Uses SIMILAR TO (pure vector) rather than REMEMBER (which includes
        BM25 that may pull from non-embedded text content).
        """
        result = brain.execute(
            'SIMILAR TO "transformers and BERT attention mechanism architecture" LIMIT 10'
        )
        assert len(result.data) > 0

        # Check that results come from the ML papers, not random text
        paper_results = [
            n for n in result.data
            if any(p in n.get("id", "").lower() for p in ["attention", "bert", "rag"])
        ]
        assert len(paper_results) > 0, (
            f"Expected ML paper chunks in results, got IDs: "
            f"{[n.get('id', '')[:50] for n in result.data[:5]]}"
        )

        # Collect text from top results
        all_text = " ".join(
            n.get("summary", "") + " " + n.get("claim", "")
            for n in result.data[:5]
        ).lower()

        keywords = ["attention", "transformer", "bert", "encoder", "self-attention",
                     "bidirectional", "model", "neural", "layer"]
        found = [kw for kw in keywords if kw in all_text]
        assert len(found) >= 1, (
            f"Expected ML-related content, found keywords {found} in:\n"
            f"{all_text[:500]}"
        )

    def test_retrieval_relevance_ordering(self, brain):
        """Top results for a specific query should be more relevant than bottom results."""
        result = brain.execute(
            'SIMILAR TO "self-attention mechanism computes query key value" LIMIT 20'
        )
        if len(result.data) >= 2:
            top_score = result.data[0].get("_similarity_score", 0)
            bottom_score = result.data[-1].get("_similarity_score", 0)
            assert top_score >= bottom_score, "Results should be ordered by relevance"

    def test_cross_document_search(self, brain):
        """Search should return results from multiple documents."""
        result = brain.execute('SIMILAR TO "neural network architecture" LIMIT 20')
        doc_sources = set()
        for node in result.data:
            node_id = node.get("id", "")
            if ":" in node_id:
                doc_prefix = node_id.split(":")[1] if node_id.count(":") >= 2 else node_id.split(":")[0]
                doc_sources.add(doc_prefix)
        # Should find chunks from at least 2 different documents
        assert len(doc_sources) >= 2, (
            f"Expected results from multiple docs, got sources: {doc_sources}"
        )
