
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class Reranker(Protocol):

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        ...


class FlashRankReranker:

    def __init__(self, model_name: str = "rank-T5-flan", max_length: int = 512):
        try:
            from flashrank import Ranker
        except ImportError as e:
            raise ImportError(
                "FlashRankReranker requires flashrank. "
                "Install with: pip install flashrank"
            ) from e

        self._ranker = Ranker(model_name=model_name, max_length=max_length)
        self._model_name = model_name

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.empty(0, dtype=np.float64)

        from flashrank import RerankRequest

        passages = [{"id": i, "text": doc} for i, doc in enumerate(documents)]
        request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(request)
        scores = np.full(len(documents), -np.inf, dtype=np.float64)
        for r in results:
            scores[r["id"]] = r["score"]
        return scores


class OnnxReranker:

    def __init__(self, model_dir: str | Path, onnx_file: str = "onnx/model_int8.onnx", max_length: int = 512):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "OnnxReranker requires onnxruntime and tokenizers."
            ) from e

        model_dir = Path(model_dir)
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=max_length)

        from supergraph.core.compute_profile import get_profile
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = get_profile().rerank_threads
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        onnx_path = model_dir / onnx_file
        self._session = ort.InferenceSession(str(onnx_path), sess_options=sess_options)
        self._input_names = {i.name for i in self._session.get_inputs()}

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.empty(0, dtype=np.float64)

        encoded = [self._tokenizer.encode(query, doc) for doc in documents]

        max_len = max(len(e.ids) for e in encoded)
        input_ids = np.zeros((len(encoded), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(encoded), max_len), dtype=np.int64)

        for i, enc in enumerate(encoded):
            length = len(enc.ids)
            input_ids[i, :length] = enc.ids
            attention_mask[i, :length] = enc.attention_mask

        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        logits = self._session.run(None, feed)[0]
        if logits.ndim == 2:
            logits = logits[:, 0]
        return logits.astype(np.float64)


class GGUFReranker:

    def __init__(self, model_path: str, projector_path: str | None = None,
                 n_ctx: int | None = None, n_gpu_layers: int = -1):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "GGUFReranker requires llama-cpp-python. "
                "Install with: pip install llama-cpp-python"
            ) from e

        temp_model = Llama(model_path=model_path, n_ctx=1, n_gpu_layers=0, verbose=False)
        native_ctx = int(temp_model.metadata.get("llama.context_length", 2048))
        del temp_model

        actual_ctx = n_ctx if n_ctx is not None else min(native_ctx, 16384)

        actual_batch = min(actual_ctx, 2048)

        self._model = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=actual_ctx,
            n_batch=actual_batch,
            n_gpu_layers=n_gpu_layers,
            flash_attn=True,
            verbose=False,
        )

        self._proj_w1 = None
        self._proj_w2 = None
        if projector_path:
            from safetensors import safe_open
            with safe_open(projector_path, framework="numpy") as f:
                self._proj_w1 = f.get_tensor("projector.0.weight")
                self._proj_w2 = f.get_tensor("projector.2.weight")

    def _embed_and_project(self, texts: str | list[str]) -> list[np.ndarray]:
        if isinstance(texts, str):
            texts = [texts]

        results = []
        for t in texts:
            e = self._model.embed(t)
            emb = np.array(e, dtype=np.float32)
            if emb.ndim == 1:
                emb = emb[np.newaxis, :]
            elif emb.ndim == 3:
                emb = emb[0]

            if self._proj_w1 is not None:
                emb = emb @ self._proj_w1.T
                emb = np.maximum(emb, 0)
                emb = emb @ self._proj_w2.T

            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            results.append(emb / norms)

        return results


    def score(self, query: str, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.empty(0, dtype=np.float64)

        q_emb = self._embed_and_project(query)[0]
        
        d_embs = self._embed_and_project(documents)
        
        scores = np.zeros(len(documents), dtype=np.float64)
        for i, d_emb in enumerate(d_embs):
            sim_matrix = q_emb @ d_emb.T
            max_sims = np.max(sim_matrix, axis=1)
            scores[i] = float(np.sum(max_sims))
            
        return scores
