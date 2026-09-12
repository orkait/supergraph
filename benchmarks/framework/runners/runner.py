
from __future__ import annotations

import signal
import time
from datetime import datetime, timezone
from typing import Any, Callable

from ..adapters.base import MemoryAdapter
from ..datasets import BenchmarkDataset
from ..metrics import RunResult


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class _Interrupted(Exception):
    pass


def run_benchmark(
    adapter: MemoryAdapter,
    dataset: BenchmarkDataset,
    *,
    k: int = 5,
    max_records: int | None = None,
    config: dict[str, Any] | None = None,
    progress_every: int = 10,
    on_interrupt: Callable[[RunResult], None] | None = None,
    qa_eval: bool = False,
) -> RunResult:
    result = RunResult(
        system_name=adapter.name,
        system_version=adapter.version,
        benchmark=dataset.name,
        config=config or {},
        started_at=_now_iso(),
    )

    records = dataset.records
    if max_records is not None:
        records = records[:max_records]

    interrupted = False

    def _handle_signal(signum, frame):
        nonlocal interrupted
        sig_name = signal.Signals(signum).name
        print(f"\n  [{adapter.name}] caught {sig_name} - saving partial results...")
        interrupted = True

    old_sigterm = signal.signal(signal.SIGTERM, _handle_signal)
    old_sigint = signal.signal(signal.SIGINT, _handle_signal)

    result.memory.start()
    t0 = time.perf_counter()

    from ..adapters.base import QueryContext

    n_records = len(records)
    print(f"[{adapter.name}] evaluating {n_records} records from {dataset.name}")
    has_ingest_done = hasattr(adapter, "ingest_done")
    has_query_with_context = hasattr(adapter, "query_with_context")

    try:
        for i, rec in enumerate(records):
            if interrupted:
                break

            adapter.reset()
            n_sessions = len(rec.sessions)
            for si, sess in enumerate(rec.sessions):
                elapsed_s = adapter.ingest(sess)
                result.latency_ingest.add(elapsed_s * 1000)
                if n_sessions >= 20 and (si + 1) % 10 == 0:
                    print(
                        f"    record {i + 1}/{n_records} "
                        f"ingest {si + 1}/{n_sessions} sessions"
                    )
            if has_ingest_done:
                adapter.ingest_done(record_metadata=rec.question.metadata)

            if has_query_with_context:
                ctx = QueryContext(
                    question=rec.question.question,
                    category=rec.question.category,
                    metadata=rec.question.metadata,
                )
                qres = adapter.query_with_context(ctx, k=k)
            else:
                qres = adapter.query(rec.question.question, k=k)
            result.latency_query.add(qres.elapsed_ms)
            ans_sess_ids = rec.question.metadata.get("answer_session_ids") or []
            raw_nodes = qres.raw if isinstance(qres.raw, list) else None
            result.quality.add(
                gold_answers=rec.question.gold_answers,
                retrieved=qres.retrieved_memories,
                category=rec.question.category,
                k=k,
                answer_session_ids=ans_sess_ids,
                retrieved_raw=raw_nodes,
            )
            if qres.tokens_used:
                result.cost.query_tokens += qres.tokens_used

            if qa_eval and qres.retrieved_memories:
                from ..transport.llm_judge import generate_answer, judge_answer
                answer = generate_answer(rec.question.question, qres.retrieved_memories)
                correct = judge_answer(
                    rec.question.question,
                    rec.question.gold_answers,
                    answer,
                    category=rec.question.category,
                )
                if correct:
                    result.quality.llm_judge_sum += 1.0
                result.quality.llm_judge_n += 1
                cat = rec.question.category
                if cat and cat in result.quality._categories:
                    bucket = result.quality._categories[cat]
                    bucket.qa_n += 1
                    if correct:
                        bucket.qa_hits += 1

            if (i + 1) % progress_every == 0 or (i + 1) == n_records:
                result.memory.snapshot_peak()
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed if elapsed else 0
                eta_s = (n_records - (i + 1)) / rate if rate else 0
                qa_str = ""
                if result.quality.llm_judge_n > 0:
                    qa_str = f" qa={result.quality.llm_judge:.3f}"
                print(
                    f"  [{adapter.name}] {i + 1}/{n_records} "
                    f"acc={result.quality.accuracy:.3f} "
                    f"r@k={result.quality.recall_at_k:.3f}"
                    f"{qa_str} "
                    f"elapsed={elapsed:.0f}s eta={eta_s:.0f}s"
                )
    except Exception:
        interrupted = True
        raise
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)
        signal.signal(signal.SIGINT, old_sigint)

        result.memory.stop()
        result.total_elapsed_s = time.perf_counter() - t0
        result.completed_at = _now_iso()

        try:
            adapter.close()
        except Exception as e:
            print(f"  [{adapter.name}] close() raised: {type(e).__name__}: {e}")

        if interrupted and on_interrupt:
            n_done = result.quality.n_questions
            print(f"  [{adapter.name}] interrupted after {n_done}/{n_records} records")
            on_interrupt(result)
            print(f"  [{adapter.name}] partial results saved")

    return result
