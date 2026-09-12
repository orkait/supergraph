
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .adapters import get_adapter
from .datasets import load_longmemeval
from .report import write_csv, write_json, write_markdown
from .runners.runner import run_benchmark


def _is_mount(path: str) -> bool:
    try:
        with open("/proc/mounts") as f:
            mounts = f.read()
        return any(f" {path} " in line for line in mounts.splitlines())
    except (FileNotFoundError, PermissionError):
        return os.path.ismount(path)


def main() -> int:
    p = argparse.ArgumentParser(prog="docker_runner")
    p.add_argument("--system", default="supergraph")
    p.add_argument("--dataset", default="longmemeval", choices=["longmemeval"])
    p.add_argument("--data-path", default="/data/longmemeval")
    p.add_argument("--variant", default="s", choices=["s", "m", "l"])
    p.add_argument("--out-dir", default="/results")
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--start", type=int, default=0,
                   help="skip this many records from the top of the file")
    p.add_argument("--categories", default="",
                   help="comma-separated question_type filter, e.g. 'multi-session,temporal-reasoning'")
    p.add_argument("--per-category", type=int, default=None,
                   help="at most this many records per category")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--embedder", default="fastembed",
                   choices=["model2vec", "fastembed", "onnx", "installed", "gguf"])
    p.add_argument("--embedder-model", default=None)
    p.add_argument("--embedder-model-dir", default=None,
                   help="local dir for onnx embedder (tokenizer.json + onnx/*.onnx)")
    p.add_argument("--embedder-cache-dir", default=None,
                   help="supergraph registry cache root (for --embedder installed)")
    p.add_argument("--embedder-output-dims", type=int, default=None)
    p.add_argument("--embedder-max-length", type=int, default=512)
    p.add_argument("--embedder-pooling", default="mean",
                   choices=["mean", "last_token"])
    p.add_argument("--embedder-threads", type=int, default=None)
    p.add_argument("--embedder-gguf-path", default=None,
                   help="path to .gguf file for gguf embedder")
    p.add_argument("--embedder-gpu-layers", type=int, default=0,
                   help="n_gpu_layers for gguf embedder (-1 = all)")
    p.add_argument("--embedder-query-prefix", default="",
                   help="query prefix for gguf embedder")
    p.add_argument("--gpu", action="store_true",
                   help="enable onnxruntime CUDA provider for ONNX embedders")
    p.add_argument("--gpu-mem-limit-gb", type=float, default=None,
                   help="GPU VRAM limit in GB for ONNX CUDA provider")
    p.add_argument("--embed-batch-size", type=int, default=128,
                   help="batch size for deferred embeddings (lower = less VRAM)")
    p.add_argument("--ceiling-mb", type=int, default=3072)
    p.add_argument("--cache-dir", default="/cache/fastembed")
    p.add_argument("--run-tag", default="")
    p.add_argument("--qa-eval", action="store_true",
                   help="run LLM-based QA evaluation after retrieval (gemma4:31b-cloud)")
    p.add_argument("--reranker", default=None, choices=["flashrank", "onnx", "gguf"],
                   help="reranker backend (flashrank=CPU, onnx=custom, gguf=Jina v3)")
    p.add_argument("--reranker-model", default="rank-T5-flan",
                   help="flashrank model name")
    p.add_argument("--reranker-model-dir", default=None,
                   help="path to reranker model (ONNX dir or GGUF file)")
    p.add_argument("--reranker-projector-path", default=None,
                   help="path to projector.safetensors (for GGUF reranker)")
    p.add_argument("--reranker-gpu-layers", type=int, default=-1,
                   help="n_gpu_layers for GGUF reranker (-1 = all)")
    p.add_argument("--reranker-onnx-file", default="onnx/model_int8.onnx",
                   help="ONNX file within reranker model dir")
    p.add_argument("--retrieval-strategy", default=None,
                   help="adapter retrieval strategy: remember_rerank, full_rerank, remember_lexical, full, etc.")
    p.add_argument("--entity-extractor", default=None,
                   help="dsl.entity_extractor - tinybert_onnx, regex, or None")
    p.add_argument("--entity-model-dir", default=None,
                   help="dsl.entity_model_dir - local dir for ONNX entity extractor")

    p.add_argument("--retrieval-depth", type=int, default=None,
                   help="REMEMBER candidate multiplier (LIMIT k*depth)")
    p.add_argument("--recall-depth", type=int, default=None,
                   help="graph traversal depth for RECALL in multi-session queries")
    p.add_argument("--max-query-entities", type=int, default=None,
                   help="max entities extracted from query for RECALL")
    p.add_argument("--recency-boost-k", type=int, default=None,
                   help="multiplier for recency-sorted results in knowledge-update")

    p.add_argument("--remember-weights", default=None,
                   help="dsl.remember_weights - 3 or 4 comma-separated fusion weights (vec,bm25,recency[,graph])")
    p.add_argument("--search-oversample", type=int, default=None,
                   help="vector.search_oversample - usearch ANN oversample factor")
    p.add_argument("--recall-decay", type=float, default=None,
                   help="dsl.recall_decay - spreading activation decay per hop")
    p.add_argument("--similarity-threshold", type=float, default=None,
                   help="vector.similarity_threshold - minimum cosine similarity")
    p.add_argument("--fusion-method", default=None, choices=["weighted", "rrf"],
                   help="dsl.fusion_method - score fusion strategy")
    p.add_argument("--rrf-k", type=float, default=None,
                   help="dsl.rrf_k - RRF ranking constant (default 60)")
    args = p.parse_args()


    if not args.embedder_model:
        if args.embedder_model_dir:
            args.embedder_model = Path(args.embedder_model_dir).name
        elif args.embedder_gguf_path:
            args.embedder_model = Path(args.embedder_gguf_path).stem
        elif args.embedder == "installed":
            print("error: --embedder installed requires --embedder-model", file=sys.stderr)
            return 2
        else:
            args.embedder_model = args.embedder

    if args.embedder == "onnx" and args.embedder_model_dir:
        model_dir = Path(args.embedder_model_dir)
        if not model_dir.exists():
            print(f"error: embedder model dir not found: {model_dir}", file=sys.stderr)
            print("hint: did you mount the model volume? (-v /host/path:/container/path:ro)", file=sys.stderr)
            return 2

    if args.embedder == "gguf" and args.embedder_gguf_path:
        gguf_path = Path(args.embedder_gguf_path)
        if not gguf_path.exists():
            print(f"error: gguf model not found: {gguf_path}", file=sys.stderr)
            print("hint: did you mount the model volume?", file=sys.stderr)
            return 2

    if args.gpu_mem_limit_gb and not args.gpu:
        print("warning: --gpu-mem-limit-gb has no effect without --gpu", file=sys.stderr)

    out_dir = Path(args.out_dir)
    if os.path.exists("/.dockerenv") and not _is_mount(args.out_dir):
        print(f"warning: {args.out_dir} is not a mounted volume - results will be lost when container exits", file=sys.stderr)
        print(f"hint: add -v $(pwd)/results:{args.out_dir} to your docker run command", file=sys.stderr)


    if args.dataset != "longmemeval":
        print(f"unknown dataset {args.dataset}", file=sys.stderr)
        return 2

    cats = {c.strip() for c in args.categories.split(",") if c.strip()} or None
    dataset = load_longmemeval(
        args.data_path,
        variant=args.variant,
        max_records=args.max_records,
        start=args.start,
        categories=cats,
        per_category=args.per_category,
    )
    print(f"loaded {len(dataset)} records from {dataset.name}")
    if cats:
        print(f"categories filter: {sorted(cats)}")
    if args.per_category:
        print(f"per-category cap: {args.per_category}")
    if args.start:
        print(f"start offset: {args.start}")

    if len(dataset) == 0:
        print("warning: 0 records after filtering - nothing to benchmark", file=sys.stderr)
        return 0


    adapter_cls = get_adapter(args.system)
    adapter_config = {
        "embedder": args.embedder,
        "embedder_model": args.embedder_model,
        "embedder_model_dir": args.embedder_model_dir,
        "embedder_cache_dir": args.embedder_cache_dir,
        "embedder_output_dims": args.embedder_output_dims,
        "embedder_max_length": args.embedder_max_length,
        "embedder_pooling": args.embedder_pooling,
        "embedder_threads": args.embedder_threads,
        "embedder_gpu": args.gpu,
        "embedder_gpu_mem_limit": int(args.gpu_mem_limit_gb * 1024**3) if args.gpu_mem_limit_gb else None,
        "embedder_gguf_path": args.embedder_gguf_path,
        "embedder_gpu_layers": args.embedder_gpu_layers,
        "embedder_query_prefix": args.embedder_query_prefix,
        "embed_batch_size": args.embed_batch_size,
        "ceiling_mb": args.ceiling_mb,
        "cache_dir": args.cache_dir,
        "reranker": args.reranker,
        "reranker_model": args.reranker_model,
        "reranker_model_dir": args.reranker_model_dir,
        "reranker_onnx_file": args.reranker_onnx_file,
        "reranker_projector_path": args.reranker_projector_path,
    }
    for attr, key in [
        ("retrieval_depth", "retrieval_depth"),
        ("recall_depth", "recall_depth"),
        ("max_query_entities", "max_query_entities"),
        ("recency_boost_k", "recency_boost_k"),
        ("search_oversample", "search_oversample"),
        ("recall_decay", "recall_decay"),
        ("similarity_threshold", "similarity_threshold"),
        ("fusion_method", "fusion_method"),
        ("rrf_k", "rrf_k"),
        ("retrieval_strategy", "retrieval_strategy"),
        ("entity_extractor", "entity_extractor"),
        ("entity_model_dir", "entity_model_dir"),
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            adapter_config[key] = val

    if "entity_model_dir" in adapter_config and adapter_config["entity_model_dir"]:
        ent_dir = Path(adapter_config["entity_model_dir"])
        if not ent_dir.exists():
            print(f"ERROR: entity_model_dir not found: {ent_dir}")
            sys.exit(1)
        print(f"✓ entity_model_dir: {ent_dir}")
    if args.remember_weights:
        adapter_config["remember_weights"] = [float(w) for w in args.remember_weights.split(",")]
    adapter = adapter_cls(config=adapter_config)
    print(f"system: {adapter.name} v{adapter.version}")
    print(f"config: {adapter_config}")

    serializable_config = {k: v for k, v in adapter_config.items() if not k.startswith("_")}


    def _save_results(result, tag_suffix=""):
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = result.started_at.replace(":", "-")[:19]
        tag = f"_{args.run_tag}" if args.run_tag else ""
        n_part = f"_n{args.max_records}" if args.max_records else "_full"
        prefix = out_dir / (
            f"{adapter.name}_{args.dataset}_{args.variant}{n_part}{tag}{tag_suffix}_{stamp}"
        )
        write_json([result], prefix.with_suffix(".json"))
        write_csv([result], prefix.with_suffix(".csv"))
        write_markdown([result], prefix.with_suffix(".md"))
        return prefix

    result = run_benchmark(
        adapter, dataset,
        k=args.k,
        max_records=args.max_records,
        config=serializable_config,
        progress_every=10,
        on_interrupt=lambda partial: _save_results(partial, "_partial"),
        qa_eval=args.qa_eval,
    )

    prefix = _save_results(result)

    print()
    print("RESULTS")
    print(f"  retrieval acc  {result.quality.accuracy:.3f}")
    print(f"  retrieval r@5  {result.quality.recall_at_k:.3f}")
    if result.quality.llm_judge_n > 0:
        print(f"  QA accuracy    {result.quality.llm_judge:.3f} (overall, n={result.quality.llm_judge_n})")
        qa_cats = [b for b in result.quality._categories.values() if b.qa_n > 0]
        if qa_cats:
            task_avg = sum(b.qa_accuracy for b in qa_cats) / len(qa_cats)
            print(f"  QA task-avg    {task_avg:.3f} (mean of {len(qa_cats)} categories)")
    print(f"  query p50      {result.latency_query.p50:.1f} ms")
    print(f"  query p95      {result.latency_query.p95:.1f} ms")
    print(f"  query p99      {result.latency_query.p99:.1f} ms")
    print(f"  ingest mean    {result.latency_ingest.mean:.1f} ms")
    print(f"  peak RSS       {result.memory.rss_peak_mb:.0f} MB")
    print(f"  elapsed        {result.total_elapsed_s:.0f} s")
    if result.quality._categories:
        print("  by category:")
        has_qa = any(b.qa_n > 0 for b in result.quality._categories.values())
        for cat, b in sorted(result.quality._categories.items()):
            qa_str = f" qa={b.qa_accuracy:.3f}" if has_qa and b.qa_n > 0 else ""
            print(f"    {cat:<30} n={b.n:<3} acc={b.accuracy:.3f} r@k={b.recall_at_k:.3f}{qa_str}")
    print()
    print(f"results: {prefix}.{{json,csv,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
