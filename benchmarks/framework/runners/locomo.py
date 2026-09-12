
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

from ..adapters.base import QueryContext, TimedOperation
from ..datasets import load_locomo
from ..transport.llm_client import (
    compute_f1, compute_llm_judge, health_check,
)

_CAT_TO_ID = {
    "multi-hop": 1, "single-hop": 2, "temporal": 3,
    "open-domain": 4, "adversarial": 5,
}

_CAT_ORDER = ["open-domain", "single-hop", "multi-hop", "temporal", "adversarial"]


_LOCOMO_QA_PROMPT = """\
You have memory of a user's conversations. A hybrid retrieval engine
(semantic + keyword + recency + graph) returned the top-{k} items most
relevant to the question. Each item is one fact from one session,
prefixed with [date] speaker:.

Facts may interleave across items. Any single item is often incomplete.
Your job: cross-reference across items to assemble the answer. The
session [date] on one item grounds relative words ("yesterday",
"last week") in that same item's text. Overlapping names or events
between items let you link claims.

Think silently through these steps:
  1. Which items bear on the question?
  2. Link overlapping entities/events across items.
  3. Resolve relative dates using the session [date] in the same item.
  4. If no combination of items supports a precise answer, abstain.

Then output ONLY the answer. Rules:
- Minimum words. One phrase. No explanation, no hedging, no quotes,
  no citation markers.
- If the question asks for multiple things, return a comma-separated list.
- If asked WHEN, output a date as "D Month YYYY" (or "YYYY" alone).
- If the retrieved items do NOT support a precise answer, output
  EXACTLY: "no information available". Do not guess, infer, or
  default to a session date.
- Prefer the exact wording from the context for names and places.

Retrieved items (top-{k}, by combined retrieval score):
{context}

Question: {question}

Answer:"""


def _build_locomo_prompt(question: str, retrieved: list[str]) -> str:
    context = "\n\n".join(f"[{j + 1}]: {t}" for j, t in enumerate(retrieved))
    return _LOCOMO_QA_PROMPT.format(
        k=len(retrieved) if retrieved else 0,
        context=context if context else "(no retrieved context)",
        question=question,
    )


def run_locomo(
    adapter,
    dataset,
    k: int = 5,
    max_questions: int | None = None,
    reranker=None,
    llm_workers: int = 8,
    judge: str = "token-f1",
    verbose: bool = False,
) -> dict:
    health_check()

    conversations: dict[str, list] = defaultdict(list)
    sessions_by_conv: dict[str, list] = {}

    for rec in dataset.records:
        conv_id = rec.question.metadata.get("sample_id", "unknown")
        conversations[conv_id].append(rec)
        if conv_id not in sessions_by_conv:
            sessions_by_conv[conv_id] = rec.sessions

    results_by_category: dict[str, list[float]] = defaultdict(list)
    all_f1: list[float] = []
    all_details: list[dict] = []
    total_ingest_ms = 0
    total_query_ms = 0
    q_count = 0

    n_convs = len(conversations)
    for conv_idx, (conv_id, records) in enumerate(conversations.items()):
        qas = records
        if max_questions is not None:
            qas = qas[:max_questions]

        sessions = sessions_by_conv[conv_id]
        print(f"\n[{conv_idx+1}/{n_convs}] [{conv_id}] Ingesting {len(sessions)} sessions, {len(qas)} questions...")

        adapter.reset()
        t0 = time.perf_counter()
        has_ingest_done = hasattr(adapter, "ingest_done")
        for si, sess in enumerate(sessions):
            adapter.ingest(sess)
            if (si + 1) % 5 == 0:
                print(f"  ingest {si+1}/{len(sessions)} sessions")

        if has_ingest_done:
            adapter.ingest_done()
        ingest_ms = (time.perf_counter() - t0) * 1000
        total_ingest_ms += ingest_ms
        print(f"  Ingested in {ingest_ms:.0f}ms")

        print(f"[{conv_id}] Retrieving {len(qas)} questions...")
        has_query_ctx = hasattr(adapter, "query_with_context")

        retrieval_results = []
        for i, rec in enumerate(qas):
            with TimedOperation() as t:
                if has_query_ctx:
                    ctx = QueryContext(
                        question=rec.question.question,
                        category=rec.question.category,
                    )
                    qres = adapter.query_with_context(ctx, k=k)
                else:
                    qres = adapter.query(rec.question.question, k=k)

                if reranker and len(qres.retrieved_memories) > k:
                    scores = reranker.score(rec.question.question, qres.retrieved_memories)
                    ranked = sorted(zip(scores, qres.retrieved_memories), reverse=True)
                    qres.retrieved_memories = [t for _, t in ranked[:k]]

            total_query_ms += t.elapsed_ms
            retrieval_results.append(qres)

        from ..transport.llm_runner import get_shared_runner
        import asyncio

        runner = get_shared_runner()
        print(
            f"[{conv_id}] Retrieval done ({total_query_ms:.0f}ms). "
            f"Generating {len(qas)} answers (rpm={runner.rpm or 'unlimited'})..."
        )

        prompts = [
            _build_locomo_prompt(
                question=rec.question.question,
                retrieved=retrieval_results[i].retrieved_memories,
            )
            for i, rec in enumerate(qas)
        ]

        def _on_progress(done: int, total: int) -> None:
            if done == total or done % 20 == 0:
                print(f"    answers {done}/{total}", flush=True)

        answers = asyncio.run(runner.call_many(prompts, max_tokens=1000, on_progress=_on_progress))
        answered_total = sum(1 for a in answers if a)
        print(f"  [{conv_id}] LLM complete: {answered_total}/{len(qas)} answered")

        want_token = judge in ("token-f1", "both")
        want_llm = judge in ("llm", "both")
        for i, rec in enumerate(qas):
            answer = answers[i] or ""
            gold = rec.question.gold_answers[0] if rec.question.gold_answers else ""
            cat_id = _CAT_TO_ID.get(rec.question.category)
            token_f1 = compute_f1(answer, gold, category=cat_id) if want_token else None
            judge_score = (
                compute_llm_judge(answer, gold, rec.question.question, category=cat_id)
                if want_llm else None
            )
            primary = judge_score if judge == "llm" else token_f1

            all_f1.append(primary)
            results_by_category[rec.question.category].append(primary)
            detail = {
                "conversation": conv_id,
                "question": rec.question.question,
                "gold": gold,
                "answer": answer,
                "category": rec.question.category,
                "category_id": cat_id,
                "retrieved": retrieval_results[i].retrieved_memories[:3],
            }
            if token_f1 is not None:
                detail["token_f1"] = round(token_f1, 4)
            if judge_score is not None:
                detail["llm_judge"] = round(judge_score, 4)
            detail["f1"] = round(primary, 4)
            all_details.append(detail)
            q_count += 1
            if verbose:
                cat = rec.question.category or "?"
                ret0 = retrieval_results[i].retrieved_memories[0] if retrieval_results[i].retrieved_memories else ""
                tf = f"{token_f1:.2f}" if token_f1 is not None else "-"
                jf = f"{judge_score:.2f}" if judge_score is not None else "-"
                print(f"  Q{i+1:>3} [{cat:<12}] tok={tf} llm={jf} "
                      f"gold={gold!r:<45} pred={(answer or '')[:80]!r}")
                print(f"         ret[0]: {ret0[:100]!r}")

        conv_f1 = sum(all_f1[-len(qas):]) / len(qas) if qas else 0
        print(f"  [{conv_id}] {len(qas)} Qs, conv_f1={conv_f1:.3f}, running_avg={sum(all_f1)/len(all_f1):.3f}")

    overall_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0

    by_category = {}
    for cat in _CAT_ORDER:
        scores = results_by_category.get(cat, [])
        if scores:
            by_category[cat] = {
                "n": len(scores),
                "f1": round(sum(scores) / len(scores), 4),
                "category_id": _CAT_TO_ID.get(cat),
            }

    summary = {
        "benchmark": "LoCoMo",
        "n_conversations": len(conversations),
        "n_questions": len(all_f1),
        "overall_f1": round(overall_f1, 4),
        "by_category": by_category,
        "ingest_ms": round(total_ingest_ms, 1),
        "query_avg_ms": round(total_query_ms / max(q_count, 1), 1),
        "adapter": adapter.name,
        "judge": judge,
    }
    if judge == "both":
        tok_vals = [d["token_f1"] for d in all_details if "token_f1" in d]
        llm_vals = [d["llm_judge"] for d in all_details if "llm_judge" in d]
        if tok_vals:
            summary["overall_token_f1"] = round(sum(tok_vals) / len(tok_vals), 4)
        if llm_vals:
            summary["overall_llm_judge"] = round(sum(llm_vals) / len(llm_vals), 4)

    return summary, all_details


def main():
    parser = argparse.ArgumentParser(prog="run_locomo")
    parser.add_argument("--data-path", default="/tmp/locomo")
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None,
                        help="max questions PER conversation")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--embedder", default="model2vec")
    parser.add_argument("--adapter", default="supergraph",
                        choices=["supergraph", "skill", "bonsai"],
                        help="supergraph = deterministic NER+CREATE; "
                             "skill = LLM-driven DSL emission via supergraph-dsl skill; "
                             "bonsai = local Ternary-Bonsai 4B TQ1_0 for ingest + recall")
    parser.add_argument("--skill-dump-dir", default=None,
                        help="Only used with --adapter skill: dump raw LLM output per session")
    parser.add_argument("--no-carry-facts", action="store_true",
                        help="Only with --adapter skill: disable cross-session fact memory")
    parser.add_argument("--bonsai-gpu-layers", type=int, default=0,
                        help="Only with --adapter bonsai: -1 offloads all layers, 0 CPU only")
    parser.add_argument("--bonsai-prompt", default=None,
                        help="Only with --adapter bonsai: override prompt file (default: lite)")
    parser.add_argument("--bonsai-kv-cache", default=None,
                        help="Only with --adapter bonsai: persistent KV cache file")
    parser.add_argument("--use-raw-turns", action="store_true",
                        help="Feed raw dialogue turns (~20/session) instead of "
                             "author-distilled observations (~9/session). Required "
                             "for fair A/B of LLM-ingest adapters.")
    parser.add_argument("--judge", default="token-f1",
                        choices=["token-f1", "llm", "both"],
                        help="Scoring metric. token-f1 = snap-research official. "
                             "llm = LLM-as-judge (semantic equivalence, matches "
                             "Mem0/Zep publish style, costs +1 LLM call per QA). "
                             "both = compute + report both.")
    parser.add_argument("--out-dir", default="benchmarks/framework/results")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-question gold/pred/F1/judge/retrieved[0].")
    args = parser.parse_args()

    ds = load_locomo(
        args.data_path,
        max_conversations=args.max_conversations,
        use_raw_turns=args.use_raw_turns,
    )
    print(f"LoCoMo: {len(ds)} total QA pairs, {len(set(r.question.metadata.get('sample_id') for r in ds.records))} conversations")

    config = {"ceiling_mb": 512}
    if ":" in args.embedder:
        backend, model = args.embedder.split(":", 1)
        config["embedder"] = backend
        config["embedder_model"] = model
    else:
        config["embedder"] = args.embedder

    if args.adapter == "skill":
        from ..adapters.supergraph_skill import SuperGraphSkillAdapter
        if args.skill_dump_dir:
            config["skill_dump_raw_dir"] = args.skill_dump_dir
        if args.no_carry_facts:
            config["skill_carry_facts"] = False
        adapter = SuperGraphSkillAdapter(config=config)
    elif args.adapter == "bonsai":
        from ..adapters.supergraph_bonsai import SuperGraphBonsaiAdapter
        config["bonsai_n_gpu_layers"] = args.bonsai_gpu_layers
        if args.bonsai_prompt:
            config["bonsai_prompt_path"] = args.bonsai_prompt
        if args.bonsai_kv_cache:
            config["bonsai_kv_cache_path"] = args.bonsai_kv_cache
        config["bonsai_n_ctx"] = 4096
        adapter = SuperGraphBonsaiAdapter(config=config)
    else:
        from ..adapters.supergraph_ import SuperGraphAdapter
        adapter = SuperGraphAdapter(config=config)

    summary, details = run_locomo(
        adapter, ds, k=args.k, max_questions=args.max_questions,
        judge=args.judge, verbose=args.verbose,
    )

    print(f"\n{'='*60}")
    print(f"LOCOMO RESULTS")
    print(f"  System:      {summary['adapter']}")
    print(f"  Convs:       {summary['n_conversations']}")
    print(f"  Questions:   {summary['n_questions']}")
    print(f"  Judge:       {summary['judge']}")
    print(f"  Overall:     {summary['overall_f1']:.4f}")
    if summary.get("overall_token_f1") is not None and summary["judge"] == "both":
        print(f"    token F1:  {summary['overall_token_f1']:.4f}")
        print(f"    LLM judge: {summary['overall_llm_judge']:.4f}")
    print(f"  By category (official order):")
    for cat, v in summary["by_category"].items():
        print(f"    {cat:<20} (cat-{v['category_id']}) n={v['n']:<4} score={v['f1']:.4f}")
    print(f"  Ingest:      {summary['ingest_ms']:.0f}ms total")
    print(f"  Query avg:   {summary['query_avg_ms']:.1f}ms")
    print(f"{'='*60}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"locomo_{summary['adapter']}.json"
    out_path.write_text(json.dumps({"summary": summary, "details": details}, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
