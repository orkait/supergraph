
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LatencyMetrics:
    samples_ms: list[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.samples_ms.append(ms)

    def _arr(self) -> np.ndarray:
        return np.asarray(self.samples_ms, dtype=np.float64)

    @property
    def p50(self) -> float:
        return float(np.percentile(self._arr(), 50)) if self.samples_ms else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self._arr(), 95)) if self.samples_ms else 0.0

    @property
    def p99(self) -> float:
        return float(np.percentile(self._arr(), 99)) if self.samples_ms else 0.0

    @property
    def mean(self) -> float:
        return float(self._arr().mean()) if self.samples_ms else 0.0

    @property
    def stddev(self) -> float:
        return float(self._arr().std(ddof=1)) if len(self.samples_ms) >= 2 else 0.0

    def to_dict(self) -> dict:
        return {
            "p50_ms": round(self.p50, 2),
            "p95_ms": round(self.p95, 2),
            "p99_ms": round(self.p99, 2),
            "mean_ms": round(self.mean, 2),
            "stddev_ms": round(self.stddev, 2),
            "n": len(self.samples_ms),
        }


@dataclass
class MemoryMetrics:
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    rss_peak_mb: float = 0.0
    _available: bool = True

    def _current_rss_mb(self) -> float:
        try:
            import psutil
        except ImportError:
            self._available = False
            return 0.0
        return psutil.Process(os.getpid()).memory_info().rss / 1_000_000

    def start(self) -> None:
        self.rss_before_mb = self._current_rss_mb()
        self.rss_peak_mb = self.rss_before_mb

    def snapshot_peak(self) -> None:
        now = self._current_rss_mb()
        if now > self.rss_peak_mb:
            self.rss_peak_mb = now

    def stop(self) -> None:
        self.rss_after_mb = self._current_rss_mb()
        self.snapshot_peak()

    @property
    def delta_mb(self) -> float:
        return self.rss_after_mb - self.rss_before_mb

    def to_dict(self) -> dict:
        return {
            "rss_before_mb": round(self.rss_before_mb, 1),
            "rss_after_mb": round(self.rss_after_mb, 1),
            "rss_peak_mb": round(self.rss_peak_mb, 1),
            "delta_mb": round(self.delta_mb, 1),
            "psutil_available": self._available,
        }


@dataclass
class _CategoryBucket:
    n: int = 0
    hits: int = 0
    recall_sum: float = 0.0
    qa_hits: int = 0
    qa_n: int = 0

    @property
    def accuracy(self) -> float:
        return self.hits / self.n if self.n else 0.0

    @property
    def recall_at_k(self) -> float:
        return self.recall_sum / self.n if self.n else 0.0

    @property
    def qa_accuracy(self) -> float:
        return self.qa_hits / self.qa_n if self.qa_n else 0.0


@dataclass
class QualityMetrics:

    n_questions: int = 0
    n_hits: int = 0
    recall_at_k_sum: float = 0.0
    llm_judge_sum: float = 0.0
    llm_judge_n: int = 0
    _categories: dict[str, _CategoryBucket] = field(default_factory=dict)

    def add(
        self,
        *,
        gold_answers: list[str],
        retrieved: list[str],
        k: int = 5,
        category: str | None = None,
        llm_judge_score: float | None = None,
        answer_session_ids: list[str] | None = None,
        retrieved_raw: list[dict] | None = None,
    ) -> None:
        gold_lower = [g.strip().lower() for g in gold_answers if g]
        if not gold_lower:
            return
        self.n_questions += 1
        retrieved_trunc_texts = [r.strip().lower() for r in retrieved[:k]]
        substring_hits = sum(
            1 for g in gold_lower
            if any(g in r for r in retrieved_trunc_texts)
        )

        session_hit = False
        if answer_session_ids and retrieved_raw:
            ans_set = set(answer_session_ids)
            for node in retrieved_raw[:k]:
                sess = (node.get("session") or "").strip()
                if not sess:
                    continue
                for ans_id in ans_set:
                    if ans_id in sess:
                        session_hit = True
                        break
                if session_hit:
                    break

        hit = 1 if (substring_hits > 0 or session_hit) else 0
        recall = substring_hits / len(gold_lower)
        if session_hit and recall < 1.0:
            recall = max(recall, 1.0 / len(gold_lower))

        self.n_hits += hit
        self.recall_at_k_sum += recall

        if category is not None:
            bucket = self._categories.setdefault(category, _CategoryBucket())
            bucket.n += 1
            bucket.hits += hit
            bucket.recall_sum += recall

        if llm_judge_score is not None:
            self.llm_judge_sum += llm_judge_score
            self.llm_judge_n += 1

    @property
    def accuracy(self) -> float:
        return self.n_hits / self.n_questions if self.n_questions else 0.0

    @property
    def recall_at_k(self) -> float:
        return self.recall_at_k_sum / self.n_questions if self.n_questions else 0.0

    @property
    def llm_judge(self) -> float:
        return self.llm_judge_sum / self.llm_judge_n if self.llm_judge_n else 0.0

    @property
    def llm_judge_task_averaged(self) -> float:
        qa_cats = [b for b in self._categories.values() if b.qa_n > 0]
        if not qa_cats:
            return 0.0
        return sum(b.qa_accuracy for b in qa_cats) / len(qa_cats)

    def to_dict(self) -> dict:
        by_cat = {
            cat: {
                "n": b.n,
                "accuracy": round(b.accuracy, 4),
                "recall_at_k": round(b.recall_at_k, 4),
                **({"qa_accuracy": round(b.qa_accuracy, 4), "qa_n": b.qa_n} if b.qa_n > 0 else {}),
            }
            for cat, b in sorted(self._categories.items())
        }
        return {
            "n_questions": self.n_questions,
            "accuracy": round(self.accuracy, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "llm_judge": round(self.llm_judge, 4) if self.llm_judge_n else None,
            "llm_judge_task_averaged": round(self.llm_judge_task_averaged, 4) if self.llm_judge_n else None,
            "by_category": by_cat or None,
        }


@dataclass
class CostMetrics:
    ingest_tokens: int = 0
    query_tokens: int = 0
    llm_api_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.ingest_tokens + self.query_tokens

    def to_dict(self) -> dict:
        return {
            "ingest_tokens": self.ingest_tokens,
            "query_tokens": self.query_tokens,
            "total_tokens": self.total_tokens,
            "llm_api_calls": self.llm_api_calls,
        }


@dataclass
class RunResult:

    system_name: str
    system_version: str
    benchmark: str
    config: dict[str, Any] = field(default_factory=dict)
    latency_ingest: LatencyMetrics = field(default_factory=LatencyMetrics)
    latency_query: LatencyMetrics = field(default_factory=LatencyMetrics)
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    cost: CostMetrics = field(default_factory=CostMetrics)
    total_elapsed_s: float = 0.0
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict:
        return {
            "system": self.system_name,
            "version": self.system_version,
            "benchmark": self.benchmark,
            "config": self.config,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_elapsed_s": round(self.total_elapsed_s, 2),
            "latency_ingest": self.latency_ingest.to_dict(),
            "latency_query": self.latency_query.to_dict(),
            "memory": self.memory.to_dict(),
            "quality": self.quality.to_dict(),
            "cost": self.cost.to_dict(),
        }
