
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters.base import Message, Session


@dataclass
class BenchmarkQuestion:
    question: str
    gold_answers: list[str]
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkRecord:

    question: BenchmarkQuestion
    sessions: list[Session]


@dataclass
class BenchmarkDataset:
    name: str
    records: list[BenchmarkRecord]

    def __len__(self) -> int:
        return len(self.records)


def load_longmemeval(
    data_path: str | Path,
    variant: str = "s",
    max_records: int | None = None,
    start: int = 0,
    categories: set[str] | list[str] | None = None,
    per_category: int | None = None,
) -> BenchmarkDataset:
    p = Path(data_path)
    candidates = [
        p / f"longmemeval_{variant}_cleaned.json",
        p / f"longmemeval_{variant}.json",
    ]
    data_file = next((c for c in candidates if c.exists()), None)
    if data_file is None:
        raise FileNotFoundError(
            f"LongMemEval data not found. Looked in: {', '.join(str(c) for c in candidates)}. "
            f"Download from https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
        )

    with open(data_file) as f:
        raw_records = json.load(f)

    if start > 0:
        raw_records = raw_records[start:]

    cat_filter: set[str] | None = set(categories) if categories else None
    if cat_filter is not None:
        raw_records = [r for r in raw_records if r.get("question_type") in cat_filter]

    if per_category is not None:
        by_cat: dict[str, list] = {}
        sampled: list = []
        for r in raw_records:
            cat = r.get("question_type") or "unknown"
            bucket = by_cat.setdefault(cat, [])
            if len(bucket) < per_category:
                bucket.append(r)
                sampled.append(r)
        raw_records = sampled

    if max_records is not None:
        raw_records = raw_records[:max_records]

    records: list[BenchmarkRecord] = []
    for rec in raw_records:
        raw_answer = rec.get("answer", "")
        if isinstance(raw_answer, list):
            gold = [str(a) for a in raw_answer]
        else:
            gold = [str(raw_answer)]

        haystack_sessions = rec.get("haystack_sessions", [])
        haystack_ids = rec.get("haystack_session_ids", [])
        haystack_dates = rec.get("haystack_dates", [])
        question_date = rec.get("question_date")

        sessions: list[Session] = []
        for idx, (sid, msg_list) in enumerate(zip(haystack_ids, haystack_sessions)):
            base = sid or f"sess_{idx}"
            session_id = f"h{idx:03d}_{base}"
            sess_date = haystack_dates[idx] if idx < len(haystack_dates) else None
            msgs = [
                Message(role=m.get("role", "user"), content=m.get("content", ""))
                for m in msg_list
            ]
            sessions.append(Session(
                session_id=session_id,
                messages=msgs,
                metadata={
                    "date": sess_date,
                    "position": idx,
                    "question_date": question_date,
                },
            ))

        question = BenchmarkQuestion(
            question=rec["question"],
            gold_answers=gold,
            category=rec.get("question_type"),
            metadata={
                "question_id": rec.get("question_id"),
                "question_date": question_date,
                "answer_session_ids": rec.get("answer_session_ids", []),
            },
        )
        records.append(BenchmarkRecord(question=question, sessions=sessions))

    return BenchmarkDataset(
        name=f"LongMemEval-{variant.upper()}",
        records=records,
    )


def load_locomo(
    data_path: str | Path,
    max_conversations: int | None = None,
    max_questions: int | None = None,
    use_raw_turns: bool = False,
) -> BenchmarkDataset:
    p = Path(data_path)
    candidates = [
        p / "locomo10.json",
        p / "raw" / "locomo10.json",
    ]
    data_file = next((c for c in candidates if c.exists()), None)
    if data_file is None:
        raise FileNotFoundError(
            f"LoCoMo data not found. Looked in: {', '.join(str(c) for c in candidates)}. "
            f"Download from https://huggingface.co/datasets/Percena/locomo-mc10"
        )

    with open(data_file) as f:
        raw = json.load(f)

    if max_conversations is not None:
        raw = raw[:max_conversations]

    cat_names = {1: "multi-hop", 2: "single-hop", 3: "temporal", 4: "open-domain", 5: "adversarial"}

    records: list[BenchmarkRecord] = []
    for conv in raw:
        conversation = conv["conversation"]
        speaker_a = conversation.get("speaker_a", "A")
        speaker_b = conversation.get("speaker_b", "B")

        sessions: list[Session] = []

        if not use_raw_turns:
            observations = conv.get("observation", {})
            sess_idx = 1
            while f"session_{sess_idx}_observation" in observations:
                obs_key = f"session_{sess_idx}_observation"
                date = conversation.get(f"session_{sess_idx}_date_time", "")
                msgs = []
                for speaker, facts in observations[obs_key].items():
                    for fact_text, evidence_id in facts:
                        msgs.append(Message(
                            role=speaker,
                            content=f"[{date}] {speaker}: {fact_text}",
                        ))
                sessions.append(Session(
                    session_id=f"s{sess_idx}",
                    messages=msgs,
                    metadata={"date": date, "position": sess_idx},
                ))
                sess_idx += 1

        if not sessions:
            sess_idx = 1
            while f"session_{sess_idx}" in conversation:
                turns = conversation[f"session_{sess_idx}"]
                date = conversation.get(f"session_{sess_idx}_date_time", "")
                msgs = [
                    Message(
                        role=t.get("speaker", "user"),
                        content=f"[{date}] {t.get('speaker', 'unknown')}: {t.get('text', '')}",
                    )
                    for t in turns
                ]
                sessions.append(Session(
                    session_id=f"s{sess_idx}",
                    messages=msgs,
                    metadata={"date": date, "position": sess_idx},
                ))
                sess_idx += 1

        for qa in conv["qa"]:
            cat_id = qa.get("category", 0)
            category = cat_names.get(cat_id, f"cat-{cat_id}")
            gold = qa.get("answer", "")
            if isinstance(gold, list):
                gold_answers = [str(a) for a in gold]
            else:
                gold_answers = [str(gold)]

            question = BenchmarkQuestion(
                question=qa["question"],
                gold_answers=gold_answers,
                category=category,
                metadata={
                    "evidence": qa.get("evidence", []),
                    "sample_id": conv.get("sample_id"),
                },
            )
            records.append(BenchmarkRecord(question=question, sessions=sessions))

    if max_questions is not None:
        records = records[:max_questions]

    return BenchmarkDataset(name="LoCoMo", records=records)


DATASET_LOADERS = {
    "longmemeval": load_longmemeval,
    "locomo": load_locomo,
}
