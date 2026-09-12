from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# openai is only needed by _load_reader / _answer_question (reader-LLM path).
# Keep the module import-safe without the SDK so tests that just exercise
# chunking helpers don't require an optional dep.

from supergraph import SuperGraph
from supergraph.registry.installer import load_installed_embedder, set_cache_dir


ANSWER_GENERATION_FOR_RAG = """
You are an assistant that MUST answer questions using ONLY the information provided in the context below. 

STRICT INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. Do NOT use your internal knowledge

CONTEXT:
<context>

QUESTION:
<question>

ANSWER REQUIREMENTS:
- Be direct and concise
- Only output the answer to the question without any explanation 

RESPONSE:
"""


def create_chunking(messages: list, retrieval_method: str = "pair_chunk") -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    if retrieval_method == "pair_chunk":
        for batch_number, batch in enumerate(messages, start=1):
            turns = batch["turns"]
            for turn_number, turn in enumerate(turns, start=1):
                pairs = [turn[i : i + 2] for i in range(0, len(turn), 2)]
                for pair_number, pair in enumerate(pairs, start=1):
                    assistant_text = pair[1]["content"] if len(pair) > 1 else "N/A"
                    text = (
                        f"USER: {pair[0]['content']}\n\n"
                        f"ASSISTANT: {assistant_text}"
                    )
                    chunks.append(
                        {
                            "text": text,
                            "metadata": {
                                "batch_number": batch_number,
                                "turn_number": turn_number,
                                "pair_number": pair_number,
                            },
                        }
                    )
        return chunks

    if retrieval_method == "turn_chunk":
        for batch_number, batch in enumerate(messages, start=1):
            turns = batch["turns"]
            for turn_number, turn in enumerate(turns, start=1):
                text = ""
                for message in turn:
                    text += f"{message['role'].upper()}: {message['content']}\n\n"
                chunks.append(
                    {
                        "text": text,
                        "metadata": {
                            "batch_number": batch_number,
                            "turn_number": turn_number,
                        },
                    }
                )
        return chunks

    raise ValueError(f"Unsupported retrieval_method: {retrieval_method!r}")


def build_answer_payload(probing_questions: dict, answers: dict[tuple[str, int], str]) -> dict:
    payload: dict[str, list[dict[str, Any]]] = {}
    for key, questions in probing_questions.items():
        out = []
        for index, question in enumerate(questions):
            obj = dict(question)
            obj["llm_response"] = answers.get((key, index), "")
            out.append(obj)
        payload[key] = out
    return payload


def _load_reader(base_url: str | None, model_name: str, api_key: str | None):
    """Build a one-off LLMRunner targeting the user-supplied OpenAI-compat endpoint.

    BEAM runs the reader against an endpoint the user specifies on the CLI
    (not the autoresearch config), so we construct a runner from a single
    ad-hoc provider dict. Same rate-limit / retry / fallback code path as
    the other benches.
    """
    from ..transport.llm_runner import LLMRunner
    # litellm needs "openai/<model>" to route to an openai-compatible HTTP
    # endpoint when we pass base_url. Without the prefix litellm may try
    # a native provider.
    litellm_model = model_name if "/" in model_name else f"openai/{model_name}"
    provider = {
        "pid": "beam_reader",
        "litellm_model": litellm_model,
        "api_base": base_url or "",
        "api_key": api_key or "ollama",
    }
    return LLMRunner([provider])


def _answer_question(runner, model_name: str, question: str, context: str) -> str:
    base_prompt = ANSWER_GENERATION_FOR_RAG.replace("<context>", context).replace("<question>", question)
    prompts = [
        base_prompt,
        base_prompt + "\n\nIMPORTANT: Do not include reasoning. Output only the final answer.",
    ]
    budgets = [256, 768]
    for prompt, budget in zip(prompts, budgets):
        content = runner.call_sync(prompt, max_tokens=budget, temperature=0.0)
        if content:
            return content
    return ""


def _build_embedder(spec: str, cache_dir: str | None):
    if ":" not in spec:
        raise ValueError("embedder must be in '<backend>:<model>' format")
    backend, model = spec.split(":", 1)
    if backend != "installed":
        raise ValueError("Only installed:<model> embedder is supported for BEAM runner")
    if cache_dir:
        set_cache_dir(cache_dir)
    return load_installed_embedder(model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def _create_supergraph(chunks: list[dict[str, Any]], embedder_spec: str, cache_dir: str | None, ceiling_mb: int) -> SuperGraph:
    embedder = _build_embedder(embedder_spec, cache_dir)
    gs = SuperGraph(path=None, embedder=embedder, ceiling_mb=ceiling_mb)
    gs.execute(
        'SYS REGISTER NODE KIND "beam_chunk" '
        'REQUIRED text:string OPTIONAL batch_number:int, turn_number:int, pair_number:int EMBED text'
    )
    for idx, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        fields = [
            f'CREATE NODE "chunk:{idx}" kind = "beam_chunk"',
            f'text = {json.dumps(chunk["text"])}',
            f'batch_number = {int(meta.get("batch_number", 0))}',
            f'turn_number = {int(meta.get("turn_number", 0))}',
        ]
        if "pair_number" in meta:
            fields.append(f'pair_number = {int(meta["pair_number"])}')
        fields.append(f'DOCUMENT {json.dumps(chunk["text"])}')
        gs.execute(" ".join(fields))
    return gs


def _retrieve_context(gs: SuperGraph, question: str, k: int, max_chars: int = 100000) -> str:
    q = question.replace('"', '\\"')
    result = gs.execute(f'REMEMBER "{q}" LIMIT {k} WHERE kind = "beam_chunk"')
    parts: list[str] = []
    total = 0
    for node in result.data:
        text = node.get("text", "")
        if not text:
            continue
        if total + len(text) > max_chars:
            break
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts)


def run_chat(
    *,
    chat_file: str | Path,
    probing_file: str | Path,
    output_file: str | Path,
    embedder: str,
    embedder_cache_dir: str | None,
    retrieval_method: str,
    k: int,
    reader_model_name: str,
    reader_model_url: str | None,
    reader_model_api_key: str | None,
    ceiling_mb: int = 1024,
) -> Path:
    messages = json.loads(Path(chat_file).read_text())
    probing_questions = json.loads(Path(probing_file).read_text())

    chunks = create_chunking(messages, retrieval_method=retrieval_method)
    gs = _create_supergraph(chunks, embedder, embedder_cache_dir, ceiling_mb)
    client = _load_reader(reader_model_url, reader_model_name, reader_model_api_key)
    answers: dict[tuple[str, int], str] = {}
    try:
        for key, questions in probing_questions.items():
            print(f'[beam_supergraph] section={key} n={len(questions)}')
            for index, question in enumerate(questions):
                print(f'[beam_supergraph] q {key}[{index}]')
                context = _retrieve_context(gs, question["question"], k=k)
                answers[(key, index)] = _answer_question(
                    client,
                    reader_model_name,
                    question["question"],
                    context,
                )
    finally:
        gs.close()

    payload = build_answer_payload(probing_questions, answers)
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=4))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks.framework.run_beam")
    parser.add_argument("--beam-root", default="/tmp/BEAM")
    parser.add_argument("--chat-size", required=True)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, required=True)
    parser.add_argument("--retrieval-method", default="pair_chunk", choices=["pair_chunk", "turn_chunk"])
    parser.add_argument("--embedder", default="installed:jina-v5-small-retrieval")
    parser.add_argument("--embedder-cache-dir", default="/tmp/gs_models")
    parser.add_argument("--reader-model-name", required=True)
    parser.add_argument("--reader-model-url", default=None)
    parser.add_argument("--reader-model-api-key", default=None)
    parser.add_argument("--result-file-name", default="supergraph_beam_answers.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ceiling-mb", type=int, default=1024)
    args = parser.parse_args(argv)

    beam_root = Path(args.beam_root)
    chats_root = beam_root / "chats" / args.chat_size
    for idx in range(args.start_index, args.end_index):
        chat_dir = chats_root / str(idx)
        chat_file = chat_dir / "chat_trunecated.json"
        if not chat_file.exists():
            chat_file = chat_dir / "chat.json"
        probing_file = chat_dir / "probing_questions" / "probing_questions.json"
        output_dir = Path("results") / args.chat_size / str(idx)
        output_file = output_dir / args.result_file_name
        print(f"running chat {idx}: {chat_file.name}")
        run_chat(
            chat_file=chat_file,
            probing_file=probing_file,
            output_file=output_file,
            embedder=args.embedder,
            embedder_cache_dir=args.embedder_cache_dir,
            retrieval_method=args.retrieval_method,
            k=args.k,
            reader_model_name=args.reader_model_name,
            reader_model_url=args.reader_model_url,
            reader_model_api_key=args.reader_model_api_key,
            ceiling_mb=args.ceiling_mb,
        )
        print(f"saved {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
