from pathlib import Path

import numpy as np

from supergraph.embedding.base import Embedder

from benchmarks.framework.runners.longmemeval import (
    build_corpus,
    evaluate_retrieval,
    iter_scored_entries,
    load_longmemeval,
    main,
    run_benchmark,
    session_id_from_corpus_id,
)


FIXTURE = Path(__file__).parent / "fixtures" / "benchmarks" / "longmemeval_sample.json"


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
        vectors = []
        for text in texts:
            vec = np.zeros(self.dims, dtype=np.float32)
            normalized = f"  {text.lower()}  "
            for index in range(len(normalized) - 2):
                trigram = normalized[index:index + 3]
                vec[hash(trigram) % self.dims] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


def test_load_longmemeval_reads_entries():
    data = load_longmemeval(FIXTURE)
    assert len(data) == 2
    assert data[0]["question_id"] == "q1"


def test_build_corpus_session_filters_to_user_turns():
    entry = load_longmemeval(FIXTURE)[0]
    items = build_corpus(entry, granularity="session")
    assert len(items) == 2
    assert items[0].session_id == "sess_1"
    assert "I moved to Seattle last year." in items[0].text
    assert "Seattle sounds like a great move." not in items[0].text


def test_build_corpus_session_falls_back_when_no_role_labels():
    entry = {
        "haystack_sessions": [
            [
                {"content": "raw turn one with no role"},
                {"content": "raw turn two with no role"},
            ],
        ],
        "haystack_session_ids": ["sess_no_role"],
        "haystack_dates": ["2023/01/01 (Sun) 10:00"],
    }
    items = build_corpus(entry, granularity="session")
    assert len(items) == 1
    assert "raw turn one with no role" in items[0].text
    assert "raw turn two with no role" in items[0].text


def test_build_corpus_turn_keeps_session_mapping():
    entry = load_longmemeval(FIXTURE)[0]
    items = build_corpus(entry, granularity="turn")
    assert len(items) == 4
    assert items[1].turn_id == 1
    assert session_id_from_corpus_id(items[1].corpus_id) == "sess_1"


def test_build_corpus_dedupes_duplicate_session_ids():
    entry = {
        "haystack_sessions": [
            [{"role": "user", "content": "same session"}],
            [{"role": "user", "content": "same session"}],
        ],
        "haystack_session_ids": ["dup_1", "dup_1"],
        "haystack_dates": ["2023/01/01 (Sun) 10:00", "2023/01/02 (Mon) 11:00"],
    }
    items = build_corpus(entry, granularity="session")
    assert len(items) == 1
    assert items[0].corpus_id == "session:dup_1"


def test_evaluate_retrieval_scores_known_ranking():
    ranked_ids = ["sess_2", "sess_1", "sess_3"]
    recall_any, recall_all, ndcg = evaluate_retrieval(
        ranked_ids=ranked_ids,
        correct_ids={"sess_1"},
        k=2,
    )
    assert recall_any == 1.0
    assert recall_all == 1.0
    assert 0.0 < ndcg <= 1.0


def test_abstention_entries_are_skipped_by_default():
    entries = load_longmemeval(FIXTURE)
    scored = iter_scored_entries(entries, include_abstention=False, limit=None)
    assert [entry["question_id"] for entry in scored] == ["q1"]


def test_cli_smoke_runs_lexical_mode(tmp_path):
    out_path = tmp_path / "results.jsonl"
    exit_code = main([
        str(FIXTURE),
        "--mode", "lexical",
        "--limit", "1",
        "--out", str(out_path),
    ])
    assert exit_code == 0
    assert out_path.exists()
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 1


def test_run_benchmark_lexical_finds_answer_session(tmp_path):
    out_path = tmp_path / "lexical.jsonl"
    summary = run_benchmark(
        dataset_path=FIXTURE,
        mode="lexical",
        granularity="session",
        top_k=5,
        limit=1,
        include_abstention=False,
        out_path=out_path,
        embedder=None,
    )
    assert summary["scored_questions"] == 1
    assert summary["metrics"]["session"]["recall_any@5"] == 1.0


def test_run_benchmark_similar_works_with_mock_embedder(tmp_path):
    out_path = tmp_path / "similar.jsonl"
    summary = run_benchmark(
        dataset_path=FIXTURE,
        mode="similar",
        granularity="session",
        top_k=5,
        limit=1,
        include_abstention=False,
        out_path=out_path,
        embedder=MockEmbedder(),
    )
    assert summary["scored_questions"] == 1
    assert summary["metrics"]["session"]["recall_any@5"] == 1.0
