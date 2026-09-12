
import pytest
import numpy as np
from pathlib import Path

from supergraph import SuperGraph
from supergraph.embedding.base import Embedder

pytestmark = pytest.mark.slow

FIXTURES = Path(__file__).parent / "fixtures"


class MockEmbedder(Embedder):

    @property
    def name(self) -> str:
        return "mock"

    @property
    def dims(self) -> int:
        return 64

    def encode_documents(self, texts, titles=None):
        return self._encode(texts)

    def encode_queries(self, texts):
        return self._encode(texts)

    def _encode(self, texts):
        vecs = []
        for t in texts:
            seed = hash(t) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(64).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vecs.append(vec)
        return np.array(vecs, dtype=np.float32)


@pytest.fixture
def gs(tmp_path):
    store = SuperGraph(
        path=str(tmp_path / "brain"),
        embedder=MockEmbedder(),
        ceiling_mb=256,
    )
    yield store
    store.close()


@pytest.fixture
def gs_queued(tmp_path):
    store = SuperGraph(
        path=str(tmp_path / "brain"),
        embedder=MockEmbedder(),
        ceiling_mb=256,
        queued=True,
    )
    yield store
    store.close()


def _fixture_path(subdir, filename):
    p = FIXTURES / subdir / filename
    if not p.exists():
        pytest.skip(f"Fixture not found: {p}")
    return str(p)


class TestCoreGraphWithFixtures:

    def test_build_paper_citation_graph(self, gs):
        papers = [
            "attention-is-all-you-need", "bert", "rag",
            "generative-agents", "node2vec", "word2vec", "toolformer",
        ]
        for p in papers:
            gs.execute(f'CREATE NODE "paper:{p}" kind = "paper" title = "{p}"')

        gs.execute('CREATE EDGE "paper:bert" -> "paper:attention-is-all-you-need" kind = "cites"')
        gs.execute('CREATE EDGE "paper:rag" -> "paper:bert" kind = "cites"')
        gs.execute('CREATE EDGE "paper:generative-agents" -> "paper:bert" kind = "cites"')
        gs.execute('CREATE EDGE "paper:toolformer" -> "paper:attention-is-all-you-need" kind = "cites"')

        result = gs.execute('TRAVERSE FROM "paper:bert" DEPTH 2')
        assert result.count >= 1

        result = gs.execute('SHORTEST PATH FROM "paper:rag" TO "paper:attention-is-all-you-need"')
        assert result.data is not None
        assert len(result.data) == 3

        result = gs.execute('ANCESTORS OF "paper:attention-is-all-you-need" DEPTH 2')
        assert result.data["nodes"]

        result = gs.execute('COUNT NODES WHERE kind = "paper"')
        assert result.data == 7

    def test_build_book_catalog_from_text_fixtures(self, gs):
        txt_dir = FIXTURES / "text"
        if not txt_dir.exists():
            pytest.skip("text fixtures not present")
        books = [f.stem for f in sorted(txt_dir.glob("*.txt"))][:10]
        if not books:
            pytest.skip("No .txt files in fixtures/text")

        for book in books:
            gs.execute(f'CREATE NODE "book:{book}" kind = "book" title = "{book}"')

        result = gs.execute('NODES WHERE kind = "book" LIMIT 5')
        assert len(result.data) <= 5

        result = gs.execute('AGGREGATE NODES WHERE kind = "book" SELECT COUNT()')
        assert result.data[0]["COUNT()"] == len(books)

    def test_edges_from_markdown_fixtures(self, gs):
        md_dir = FIXTURES / "markdown"
        if not md_dir.exists():
            pytest.skip("markdown fixtures not present")
        docs = [f.stem for f in sorted(md_dir.glob("*.md"))]
        if not docs:
            pytest.skip("No .md files in fixtures/markdown")

        for doc in docs:
            gs.execute(f'CREATE NODE "doc:{doc}" kind = "docs" name = "{doc}"')

        for i in range(len(docs) - 1):
            gs.execute(
                f'CREATE EDGE "doc:{docs[i]}" -> "doc:{docs[i+1]}" kind = "related"'
            )

        result = gs.execute('EDGES FROM "doc:{}" WHERE kind = "related"'.format(docs[0]))
        assert result.count >= 1


class TestDSLWithFixtures:

    def test_complex_queries_on_fixture_metadata(self, gs):
        categories = {
            "text": [f.stem for f in (FIXTURES / "text").glob("*.txt")][:5]
            if (FIXTURES / "text").exists() else [],
            "markdown": [f.stem for f in (FIXTURES / "markdown").glob("*.md")][:5]
            if (FIXTURES / "markdown").exists() else [],
            "csv": [f.stem for f in (FIXTURES / "csv").glob("*.csv")][:5]
            if (FIXTURES / "csv").exists() else [],
        }
        total = sum(len(v) for v in categories.values())
        if total == 0:
            pytest.skip("No fixture files found")

        for cat, items in categories.items():
            for item in items:
                gs.execute(
                    f'CREATE NODE "doc:{cat}:{item}" kind = "document" '
                    f'category = "{cat}" title = "{item}" size = {len(item)}'
                )

        result = gs.execute('NODES WHERE category = "text" AND size > 5')
        for node in result.data:
            assert node["category"] == "text"
            assert node["size"] > 5

        result = gs.execute(
            'AGGREGATE NODES WHERE kind = "document" GROUP BY category SELECT COUNT()'
        )
        assert len(result.data) >= 1

        result = gs.execute('NODES WHERE kind = "document" ORDER BY size DESC LIMIT 3')
        assert len(result.data) <= 3
        if len(result.data) >= 2:
            assert result.data[0]["size"] >= result.data[1]["size"]

    def test_match_pattern_on_fixture_graph(self, gs):
        gs.execute('CREATE NODE "db:faiss" kind = "vectordb" name = "faiss"')
        gs.execute('CREATE NODE "db:chroma" kind = "vectordb" name = "chroma"')
        gs.execute('CREATE NODE "paper:rag" kind = "paper" name = "rag"')
        gs.execute('CREATE EDGE "paper:rag" -> "db:faiss" kind = "uses"')
        gs.execute('CREATE EDGE "paper:rag" -> "db:chroma" kind = "uses"')

        result = gs.execute('MATCH ("paper:rag") -[kind = "uses"]-> (db)')
        assert result.data["bindings"]
        assert len(result.data["bindings"]) == 2

    def test_count_and_aggregate_csv_fixtures(self, gs):
        csv_dir = FIXTURES / "csv"
        if not csv_dir.exists():
            pytest.skip("csv fixtures not present")
        datasets = [f.stem for f in sorted(csv_dir.glob("*.csv"))]
        if not datasets:
            pytest.skip("No CSV fixture files")

        for ds in datasets:
            gs.execute(
                f'CREATE NODE "dataset:{ds}" kind = "dataset" name = "{ds}" rows = {len(ds) * 10}'
            )

        result = gs.execute('COUNT NODES WHERE kind = "dataset"')
        assert result.data == len(datasets)

        result = gs.execute(
            'AGGREGATE NODES WHERE kind = "dataset" SELECT SUM(rows), MAX(rows), MIN(rows)'
        )
        assert "SUM(rows)" in result.data[0]
        assert result.data[0]["SUM(rows)"] > 0


class TestVectorSearchWithFixtures:

    def test_similar_to_text_with_mock_embedder(self, gs):
        gs.execute(
            'SYS REGISTER NODE KIND "concept" REQUIRED topic:string EMBED topic'
        )
        gs.execute(
            'CREATE NODE "c:transformers" kind = "concept" '
            'topic = "transformer attention mechanism"'
        )
        gs.execute(
            'CREATE NODE "c:cnn" kind = "concept" '
            'topic = "convolutional neural networks"'
        )
        gs.execute(
            'CREATE NODE "c:rnn" kind = "concept" '
            'topic = "recurrent neural networks"'
        )

        result = gs.execute('SIMILAR TO "attention mechanism" LIMIT 3')
        assert result.kind == "nodes"

    def test_remember_hybrid_retrieval(self, gs):
        gs.execute(
            'SYS REGISTER NODE KIND "fact" REQUIRED claim:string EMBED claim'
        )
        gs.execute(
            'CREATE NODE "f:1" kind = "fact" claim = "transformers use self attention"'
        )
        gs.execute('CREATE NODE "f:2" kind = "fact" claim = "BERT is bidirectional"')
        gs.execute('CREATE NODE "f:3" kind = "fact" claim = "GPT is autoregressive"')

        result = gs.execute('REMEMBER "attention models" LIMIT 5')
        assert result.kind == "nodes"
        if result.data:
            assert "_remember_score" in result.data[0]

    def test_similar_to_node_reference(self, gs):
        gs.execute(
            'SYS REGISTER NODE KIND "item" REQUIRED text:string EMBED text'
        )
        gs.execute('CREATE NODE "a" kind = "item" text = "hello world"')
        gs.execute('CREATE NODE "b" kind = "item" text = "hello earth"')
        gs.execute('CREATE NODE "c" kind = "item" text = "goodbye moon"')

        result = gs.execute('SIMILAR TO NODE "a" LIMIT 3')
        assert result.kind == "nodes"

    def test_remember_with_persistence(self, tmp_path):
        gs = SuperGraph(
            path=str(tmp_path / "brain"),
            embedder=MockEmbedder(),
        )
        try:
            gs.execute(
                'CREATE NODE "m1" kind = "fact" '
                'summary = "faiss supports approximate nearest neighbor search"'
            )
            gs.execute(
                'CREATE NODE "m2" kind = "fact" '
                'summary = "chroma is a vector database for AI"'
            )
            result = gs.execute('REMEMBER "vector search" LIMIT 5')
            assert result.kind == "nodes"
        finally:
            gs.close()


@pytest.mark.needs_ingest
class TestDocumentIngestion:

    def test_ingest_text_fixture(self, gs):
        path = _fixture_path("text", "art-of-war.txt")
        result = gs.execute(f'INGEST "{path}" AS "doc:artofwar" KIND "book"')
        assert result.data["doc_id"] == "doc:artofwar"
        assert result.data["chunks"] > 0

        node = gs.execute('NODE "doc:artofwar"')
        assert node.data is not None
        assert node.data["kind"] == "book"

        edges = gs.execute('EDGES FROM "doc:artofwar"')
        assert edges.count > 0

    def test_ingest_markdown_fixture(self, gs):
        path = _fixture_path("markdown", "faiss.md")
        result = gs.execute(f'INGEST "{path}" AS "doc:faiss" KIND "docs"')
        assert result.data["doc_id"] == "doc:faiss"
        assert result.data["chunks"] > 0

    def test_ingest_html_fixture(self, gs):
        path = _fixture_path("html", "transformer.html")
        result = gs.execute(f'INGEST "{path}" AS "doc:transformer-html"')
        assert result.data["doc_id"] == "doc:transformer-html"
        assert result.data["chunks"] >= 0

    def test_ingest_csv_fixture(self, gs):
        path = _fixture_path("csv", "iris.csv")
        result = gs.execute(f'INGEST "{path}" AS "doc:iris"')
        assert result.data["doc_id"] == "doc:iris"

    def test_ingest_pdf_fixture(self, gs):
        path = _fixture_path("pdf", "attention-is-all-you-need.pdf")
        try:
            result = gs.execute(f'INGEST "{path}" AS "doc:attention"')
            assert result.data["doc_id"] == "doc:attention"
            assert result.data["chunks"] > 0
            assert result.data["parser"] in (
                "pymupdf4llm", "markitdown", "docling", "text"
            )
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("pymupdf", "markitdown", "not installed", "no module")):
                pytest.skip(f"PDF parser not available: {e}")
            raise

    def test_ingest_then_lexical_search(self, tmp_path):
        path = str(FIXTURES / "text" / "metamorphosis.txt")
        if not Path(path).exists():
            pytest.skip("metamorphosis.txt not found")

        gs = SuperGraph(
            path=str(tmp_path / "brain"),
            embedder=MockEmbedder(),
        )
        try:
            gs.execute(f'INGEST "{path}" AS "doc:metamorphosis"')
            result = gs.execute('LEXICAL SEARCH "Gregor Samsa" LIMIT 5')
            assert result.kind == "nodes"
        finally:
            gs.close()

    def test_ingest_multiple_markdown_docs(self, gs):
        md_fixtures = [
            ("markdown", "faiss.md", "doc:faiss"),
            ("markdown", "chroma.md", "doc:chroma"),
            ("markdown", "qdrant.md", "doc:qdrant"),
        ]
        ingested = []
        for subdir, filename, node_id in md_fixtures:
            p = FIXTURES / subdir / filename
            if not p.exists():
                continue
            result = gs.execute(f'INGEST "{p}" AS "{node_id}" KIND "docs"')
            assert result.data["chunks"] > 0
            ingested.append(node_id)

        if not ingested:
            pytest.skip("No markdown fixtures found")

        result = gs.execute('COUNT NODES WHERE kind = "docs"')
        assert result.data == len(ingested)

    def test_ingest_all_csv_fixtures(self, gs):
        csv_dir = FIXTURES / "csv"
        if not csv_dir.exists():
            pytest.skip("csv fixtures not present")
        csv_files = list(sorted(csv_dir.glob("*.csv")))
        if not csv_files:
            pytest.skip("No CSV files in fixtures/csv")

        for f in csv_files:
            node_id = f"doc:csv:{f.stem}"
            result = gs.execute(f'INGEST "{f}" AS "{node_id}"')
            assert result.data["doc_id"] == node_id

        result = gs.execute('COUNT NODES WHERE kind = "document"')
        assert result.data >= len(csv_files)

    def test_ingest_connect_workflow(self, tmp_path):
        gs = SuperGraph(
            path=str(tmp_path / "brain"),
            embedder=MockEmbedder(),
        )
        try:
            ingested = 0
            for name in ["faiss", "chroma", "qdrant"]:
                p = FIXTURES / "markdown" / f"{name}.md"
                if not p.exists():
                    continue
                gs.execute(f'INGEST "{p}" AS "doc:{name}" KIND "docs"')
                ingested += 1

            if ingested < 2:
                pytest.skip("Need at least 2 markdown fixtures for connect test")

            result = gs.execute('SYS CONNECT THRESHOLD 0.5')
            assert result.kind == "ok"
        finally:
            gs.close()


class TestBeliefSystemWithFixtures:

    def test_assert_retract_lifecycle(self, gs):
        gs.execute(
            'ASSERT "belief:transformers-best" kind = "belief" '
            'claim = "transformers are the best architecture" '
            'CONFIDENCE 0.8 SOURCE "paper:attention"'
        )
        gs.execute(
            'ASSERT "belief:cnns-vision" kind = "belief" '
            'claim = "CNNs are best for vision" '
            'CONFIDENCE 0.9 SOURCE "paper:alexnet"'
        )

        result = gs.execute('NODES WHERE kind = "belief"')
        assert len(result.data) == 2

        gs.execute(
            'RETRACT "belief:transformers-best" '
            'REASON "ViT surpassed CNNs in vision too"'
        )

        result = gs.execute('NODES WHERE kind = "belief"')
        assert len(result.data) == 1
        assert result.data[0]["id"] == "belief:cnns-vision"

        result = gs.execute('WHAT IF RETRACT "belief:cnns-vision"')
        assert result.kind == "counterfactual"
        assert gs.execute('NODE "belief:cnns-vision"').data is not None

    def test_recall_spreading_activation(self, gs):
        gs.execute(
            'CREATE NODE "topic:ml" kind = "topic" name = "machine learning" importance = 1.0'
        )
        gs.execute(
            'CREATE NODE "topic:dl" kind = "topic" name = "deep learning" importance = 0.9'
        )
        gs.execute(
            'CREATE NODE "topic:nlp" kind = "topic" name = "NLP" importance = 0.8'
        )
        gs.execute(
            'CREATE NODE "topic:cv" kind = "topic" name = "computer vision" importance = 0.7'
        )
        gs.execute('CREATE EDGE "topic:ml" -> "topic:dl" kind = "contains"')
        gs.execute('CREATE EDGE "topic:dl" -> "topic:nlp" kind = "contains"')
        gs.execute('CREATE EDGE "topic:dl" -> "topic:cv" kind = "contains"')

        result = gs.execute('RECALL FROM "topic:ml" DEPTH 1 LIMIT 10')
        assert result.kind == "nodes"
        names = {n.get("name") for n in result.data}
        assert "deep learning" in names

        result = gs.execute('RECALL FROM "topic:ml" DEPTH 2 LIMIT 10')
        assert result.kind == "nodes"
        names2 = {n.get("name") for n in result.data}
        assert len(names2) >= 1

    def test_propagate_field_across_graph(self, gs):
        gs.execute(
            'SYS REGISTER NODE KIND "belief" REQUIRED confidence:float'
        )
        gs.execute(
            'CREATE NODE "root-belief" kind = "belief" confidence = 0.95'
        )
        gs.execute(
            'CREATE NODE "child-belief" kind = "belief" confidence = 0.5'
        )
        gs.execute(
            'CREATE EDGE "root-belief" -> "child-belief" kind = "supports"'
        )
        result = gs.execute('PROPAGATE "root-belief" FIELD confidence DEPTH 1')
        assert result.data["updated"] >= 1

    def test_assert_upsert_semantics(self, gs):
        gs.execute(
            'ASSERT "f:evolving" kind = "fact" value = 1 CONFIDENCE 0.5'
        )
        gs.execute(
            'ASSERT "f:evolving" kind = "fact" value = 2 CONFIDENCE 0.9'
        )
        node = gs.execute('NODE "f:evolving"')
        assert node.data["value"] == 2
        assert gs.node_count == 1


class TestSystemWithFixtures:

    def test_full_lifecycle_stats_and_health(self, gs):
        for i, book in enumerate(["dracula", "frankenstein", "pride-and-prejudice"]):
            gs.execute(
                f'CREATE NODE "book:{book}" kind = "book" title = "{book}" rank = {i}'
            )

        result = gs.execute('SYS STATS')
        assert result.data["node_count"] == 3

        result = gs.execute('SYS STATUS')
        assert result.data["nodes"] == 3

        result = gs.execute('SYS HEALTH')
        assert "memory_utilization" in result.data
        assert "live_nodes" in result.data
        assert result.data["live_nodes"] == 3

        result = gs.execute('SYS LOG LIMIT 5')
        assert result.kind == "log_entries"
        assert len(result.data) >= 1

        gs.checkpoint()

    def test_snapshot_and_rollback(self, gs):
        gs.execute('CREATE NODE "s1" kind = "test" val = 1')
        gs.execute('SYS SNAPSHOT "before"')
        gs.execute('CREATE NODE "s2" kind = "test" val = 2')
        assert gs.execute('COUNT NODES').data == 2

        gs.execute('SYS ROLLBACK TO "before"')
        assert gs.execute('COUNT NODES').data == 1
        assert gs.execute('NODE "s1"').data is not None
        assert gs.execute('NODE "s2"').data is None

    def test_evict_removes_nodes(self, gs):
        for i in range(50):
            gs.execute(f'CREATE NODE "evict:{i}" kind = "temp" val = {i}')
        result = gs.execute('SYS EVICT')
        assert result.kind == "ok"

    def test_optimize_runs_successfully(self, gs):
        for i in range(10):
            gs.execute(f'CREATE NODE "opt:{i}" kind = "test" val = {i}')
        result = gs.execute('SYS OPTIMIZE')
        assert result.kind == "ok"

    @pytest.mark.needs_scheduler
    def test_cron_lifecycle(self, gs_queued):
        gs_queued.execute(
            'SYS CRON ADD "test-job" SCHEDULE "@hourly" QUERY "SYS STATS"'
        )
        result = gs_queued.execute('SYS CRON LIST')
        assert len(result.data) == 1
        assert result.data[0]["name"] == "test-job"

        gs_queued.execute('SYS CRON RUN "test-job"')

        gs_queued.execute('SYS CRON DELETE "test-job"')
        result = gs_queued.execute('SYS CRON LIST')
        assert len(result.data) == 0

    def test_log_with_trace_id(self, gs):
        gs.bind_trace("test-session")
        gs.execute('CREATE NODE "traced" kind = "test"')
        gs.discard_trace()

        result = gs.execute('SYS LOG TRACE "test-session"')
        assert len(result.data) >= 1
        assert all(e["trace_id"] == "test-session" for e in result.data)

    def test_kinds_and_describe(self, gs):
        gs.execute(
            'SYS REGISTER NODE KIND "paper" REQUIRED title:string, year:int'
        )
        result = gs.execute('SYS KINDS')
        assert "paper" in result.data

        result = gs.execute('SYS DESCRIBE NODE "paper"')
        assert result.data is not None
        assert "title" in str(result.data)


class TestAgentWorkflow:

    def test_research_agent_session(self, tmp_path):
        gs = SuperGraph(
            path=str(tmp_path / "brain"),
            embedder=MockEmbedder(),
        )
        try:
            text_path = FIXTURES / "text" / "art-of-war.txt"
            if text_path.exists():
                gs.execute(f'INGEST "{text_path}" AS "doc:artofwar" KIND "book"')

            md_path = FIXTURES / "markdown" / "faiss.md"
            if md_path.exists():
                gs.execute(f'INGEST "{md_path}" AS "doc:faiss" KIND "docs"')

            gs.execute(
                'CREATE NODE "topic:search" kind = "topic" name = "vector search"'
            )
            gs.execute(
                'CREATE NODE "topic:strategy" kind = "topic" name = "military strategy"'
            )

            if md_path.exists():
                gs.execute('CREATE EDGE "doc:faiss" -> "topic:search" kind = "about"')
            if text_path.exists():
                gs.execute(
                    'CREATE EDGE "doc:artofwar" -> "topic:strategy" kind = "about"'
                )

            if md_path.exists():
                gs.execute(
                    'ASSERT "belief:faiss-fast" kind = "belief" '
                    'claim = "FAISS is fast for similarity search" '
                    'CONFIDENCE 0.95 SOURCE "doc:faiss"'
                )

            gs.bind_trace("research-session-1")

            result = gs.execute('REMEMBER "search" LIMIT 5')
            assert result.kind == "nodes"

            result = gs.execute('RECALL FROM "topic:search" DEPTH 2 LIMIT 10')
            assert result.kind == "nodes"

            gs.discard_trace()

            gs.execute('SYS HEALTH')
            result = gs.execute('SYS LOG LIMIT 20')
            traced = [
                e for e in result.data
                if e.get("trace_id") == "research-session-1"
            ]
            assert len(traced) >= 1

            gs.checkpoint()
        finally:
            gs.close()

    def test_memory_pressure_workflow(self, gs):
        for i in range(200):
            gs.execute(
                f'CREATE NODE "mem:{i}" kind = "memory" '
                f'summary = "memory item {i}" importance = {i % 10}'
            )

        result = gs.execute('SYS HEALTH')
        assert result.data["live_nodes"] == 200

        result = gs.execute('SYS EVICT')
        assert result.kind == "ok"

        result = gs.execute('SYS OPTIMIZE')
        assert result.kind == "ok"

    def test_persistence_survives_restart(self, tmp_path):
        path = str(tmp_path / "persist")
        gs = SuperGraph(path=path, embedder=MockEmbedder())
        gs.execute('CREATE NODE "survive" kind = "test" val = 42')
        gs.checkpoint()
        gs.close()

        gs2 = SuperGraph(path=path, embedder=MockEmbedder())
        try:
            result = gs2.execute('NODE "survive"')
            assert result.data is not None
            assert result.data["val"] == 42
        finally:
            gs2.close()

    def test_batch_transaction_with_variables(self, gs):
        gs.execute(
            'BEGIN\n'
            '$a = CREATE NODE "batch:a" kind = "item" name = "alpha"\n'
            '$b = CREATE NODE "batch:b" kind = "item" name = "beta"\n'
            'CREATE EDGE $a -> $b kind = "linked"\n'
            'COMMIT'
        )
        result = gs.execute('EDGES FROM "batch:a"')
        assert len(result.data) == 1
        assert result.data[0]["target"] == "batch:b"

    def test_fixture_metadata_knowledge_graph(self, gs):
        fixture_dirs = {
            "text": "*.txt",
            "markdown": "*.md",
            "csv": "*.csv",
            "html": "*.html",
            "pdf": "*.pdf",
        }
        total_nodes = 0
        for category, pattern in fixture_dirs.items():
            d = FIXTURES / category
            if not d.exists():
                continue
            files = list(d.glob(pattern))
            for f in files[:3]:
                node_id = f"file:{category}:{f.stem}"
                gs.execute(
                    f'CREATE NODE "{node_id}" kind = "file" '
                    f'category = "{category}" name = "{f.stem}"'
                )
                total_nodes += 1

        if total_nodes == 0:
            pytest.skip("No fixture files found in any directory")

        result = gs.execute('COUNT NODES WHERE kind = "file"')
        assert result.data == total_nodes

        result = gs.execute(
            'AGGREGATE NODES WHERE kind = "file" GROUP BY category SELECT COUNT()'
        )
        assert len(result.data) >= 1

        result = gs.execute(
            'SYS STATS'
        )
        assert result.data["node_count"] == total_nodes

    def test_image_fixture_metadata_nodes(self, gs):
        img_dir = FIXTURES / "images"
        if not img_dir.exists():
            pytest.skip("images fixtures not present")
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        images = sorted(images)[:5]
        if not images:
            pytest.skip("No image files in fixtures/images")

        for img in images:
            gs.execute(
                f'CREATE NODE "img:{img.stem}" kind = "image" '
                f'name = "{img.stem}" ext = "{img.suffix.lstrip(".")}"'
            )

        result = gs.execute('COUNT NODES WHERE kind = "image"')
        assert result.data == len(images)

        result = gs.execute('NODES WHERE kind = "image"')
        assert len(result.data) == len(images)
        names = [n["name"] for n in result.data]
        assert set(names) == {img.stem for img in images}

    def test_voice_fixture_metadata_nodes(self, gs):
        voice_dir = FIXTURES / "voice"
        if not voice_dir.exists():
            pytest.skip("voice fixtures not present")
        voice_files = sorted(voice_dir.glob("*.wav")) + sorted(voice_dir.glob("*.mp3"))
        voice_files = [v for v in voice_files if v.stem.split("_")[0] in
                       ("en", "hi", "ta", "te", "mr")]
        if not voice_files:
            pytest.skip("No voice fixture files")

        for vf in voice_files:
            parts = vf.stem.split("_")
            lang = parts[0] if parts else "unknown"
            gs.execute(
                f'CREATE NODE "voice:{vf.stem}" kind = "audio" '
                f'lang = "{lang}" name = "{vf.stem}" ext = "{vf.suffix.lstrip(".")}"'
            )

        result = gs.execute('COUNT NODES WHERE kind = "audio"')
        assert result.data == len(voice_files)

        result = gs.execute(
            'AGGREGATE NODES WHERE kind = "audio" GROUP BY lang SELECT COUNT()'
        )
        assert len(result.data) >= 1
