from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9_-]{2,}(?:\s+[A-Z][a-zA-Z0-9_-]{2,}){0,3}\b")
_PREFIX_RE = re.compile(r"^\[.*?\]\s*\w+:\s*")
_BLOCKLIST = frozenset({
    "the", "a", "an",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
})
_DEFAULT_ALLOWED_LABELS = frozenset({"PER", "PERSON", "ORG", "LOC", "GPE", "MISC"})
_GENERIC_CONLL_LABELS = {
    "LABEL_0": "O",
    "LABEL_1": "B-PER",
    "LABEL_2": "I-PER",
    "LABEL_3": "B-ORG",
    "LABEL_4": "I-ORG",
    "LABEL_5": "B-LOC",
    "LABEL_6": "I-LOC",
    "LABEL_7": "B-MISC",
    "LABEL_8": "I-MISC",
}


def _strip_prefix(text: str) -> str:
    stripped = _PREFIX_RE.sub("", text or "")
    return stripped or (text or "")


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


class RegexEntityExtractor:
    def __init__(self, limit: int = 6, blocklist: frozenset[str] = _BLOCKLIST):
        self._limit = limit
        self._blocklist = blocklist

    def extract_batch(self, texts: list[str]) -> list[list[str]]:
        return [self.extract(t) for t in texts]

    def extract(self, text: str) -> list[str]:
        content = _strip_prefix(text)
        out: list[str] = []
        for raw in _ENTITY_RE.findall(content):
            norm = raw.strip()
            if len(norm) < 3:
                continue
            if norm.lower() in self._blocklist:
                continue
            out.append(norm)
        return _unique(out)[:self._limit]


class OnnxTokenClassificationEntityExtractor:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        onnx_file: str = "onnx/model_int8.onnx",
        score_threshold: float = 0.6,
        allowed_labels: set[str] | frozenset[str] = _DEFAULT_ALLOWED_LABELS,
        providers: list[str] | str | None = None,
        max_length: int = 256,
        blocklist: frozenset[str] = _BLOCKLIST,
    ):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "TinyBERT ONNX entity extraction requires onnxruntime and tokenizers"
            ) from e

        from supergraph.embedding.onnx_hf_embedder import _create_inference_session, _resolve_providers

        model_dir = Path(model_dir)
        tok_path = model_dir / "tokenizer.json"
        if not tok_path.exists():
            onnx_tok = model_dir / "onnx" / "tokenizer.json"
            if onnx_tok.exists():
                tok_path = onnx_tok
            else:
                raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")

        config_path = model_dir / "config.json"
        if not config_path.exists():
            onnx_cfg = model_dir / "onnx" / "config.json"
            if onnx_cfg.exists():
                config_path = onnx_cfg
            else:
                raise FileNotFoundError(f"config.json not found in {model_dir}")

        onnx_path = model_dir / onnx_file
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        cfg = json.loads(config_path.read_text())
        raw_id2label = cfg.get("id2label")
        if not raw_id2label:
            raise ValueError(f"id2label missing from {config_path}")
        self._id2label = {int(k): str(v) for k, v in raw_id2label.items()}
        if set(self._id2label.values()) == set(_GENERIC_CONLL_LABELS):
            self._label_aliases = dict(_GENERIC_CONLL_LABELS)
        else:
            self._label_aliases = {}

        self._tokenizer = Tokenizer.from_file(str(tok_path))
        self._tokenizer.enable_truncation(max_length=max_length)
        self._score_threshold = score_threshold
        self._allowed_labels = set(allowed_labels)
        self._blocklist = blocklist

        resolved_providers = _resolve_providers(providers)
        sess_options = ort.SessionOptions()
        self._session = _create_inference_session(
            ort,
            str(onnx_path),
            {
                "providers": resolved_providers,
                "sess_options": sess_options,
            },
        )
        active_providers = self._session.get_providers()
        print(f"  [NER] Model loaded. Provider: {active_providers[0]}")
        
        # Check strict enforcement
        if any(p in ("CUDAExecutionProvider", "TensorrtExecutionProvider") for p in resolved_providers):
            if not any(p in ("CUDAExecutionProvider", "TensorrtExecutionProvider") for p in active_providers):
                raise RuntimeError(
                    f"NER GPU provider requested but unavailable. Active: {active_providers}. "
                    "Check LD_LIBRARY_PATH and nvidia-* wheels."
                )

        self._input_names = {i.name for i in self._session.get_inputs()}
        self._needs_token_type_ids = "token_type_ids" in self._input_names

    def _decode_entities(
        self,
        text: str,
        offsets: list[tuple[int, int]],
        labels: list[str],
        scores: np.ndarray,
    ) -> list[str]:
        out: list[str] = []
        current_start: int | None = None
        current_end: int | None = None
        current_type: str | None = None
        current_scores: list[float] = []

        def flush():
            nonlocal current_start, current_end, current_type, current_scores
            if current_start is None or current_end is None or current_type is None:
                current_start = current_end = None
                current_type = None
                current_scores = []
                return
            avg_score = sum(current_scores) / max(len(current_scores), 1)
            entity = text[current_start:current_end].strip()
            if (
                entity
                and len(entity) >= 3
                and avg_score >= self._score_threshold
                and entity.lower() not in self._blocklist
            ):
                out.append(entity)
            current_start = current_end = None
            current_type = None
            current_scores = []

        aliases = getattr(self, "_label_aliases", {})
        for (start, end), label, score in zip(offsets, labels, scores):
            label = aliases.get(label, label)
            if start == end or label == "O":
                flush()
                continue

            if "-" in label:
                prefix, ent_type = label.split("-", 1)
            else:
                prefix, ent_type = "B", label

            if ent_type not in self._allowed_labels:
                flush()
                continue

            join_ok = current_end is not None and text[current_end:start].strip() == ""
            if prefix == "B" or current_type != ent_type or not join_ok:
                flush()
                current_start = start
                current_end = end
                current_type = ent_type
                current_scores = [float(score)]
                continue

            current_end = end
            current_scores.append(float(score))

        flush()
        return _unique(out)

    def extract_batch(self, texts: list[str]) -> list[list[str]]:
        if not texts:
            return []
        
        stripped_texts = [_strip_prefix(t) for t in texts]
        encodings = [self._tokenizer.encode(t) for t in stripped_texts]
        
        # Simple padding for batch inference
        max_len = max(len(e.ids) for e in encodings)
        input_ids = []
        attention_mask = []
        token_type_ids = []
        
        for e in encodings:
            pad_len = max_len - len(e.ids)
            input_ids.append(e.ids + [0] * pad_len)
            attention_mask.append(e.attention_mask + [0] * pad_len)
            if self._needs_token_type_ids:
                token_type_ids.append([0] * max_len)
                
        feed = {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "attention_mask": np.array(attention_mask, dtype=np.int64),
        }
        if self._needs_token_type_ids:
            feed["token_type_ids"] = np.array(token_type_ids, dtype=np.int64)
            
        all_logits = self._session.run(None, feed)[0]
        
        results = []
        for i, logits in enumerate(all_logits):
            probs = _softmax(logits[:len(encodings[i].ids)])
            pred_ids = np.argmax(probs, axis=-1)
            labels = [self._id2label[int(idx)] for idx in pred_ids]
            scores = np.max(probs, axis=-1)
            results.append(self._decode_entities(stripped_texts[i], list(encodings[i].offsets), labels, scores)[:6])
            
        return results

    def extract(self, text: str) -> list[str]:
        return self.extract_batch([text])[0]


def build_entity_extractor(config: dict[str, Any]):
    mode = (config.get("entity_extractor") or "regex").lower()
    if mode == "regex":
        return RegexEntityExtractor()
    if mode == "tinybert_onnx":
        model_dir = config.get("entity_model_dir")
        if not model_dir:
            raise ValueError("entity_model_dir is required for entity_extractor=tinybert_onnx")
        gpu = bool(config.get("entity_gpu", config.get("embedder_gpu", False)))
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        labels = config.get("entity_allowed_labels") or sorted(_DEFAULT_ALLOWED_LABELS)
        return OnnxTokenClassificationEntityExtractor(
            model_dir=model_dir,
            onnx_file=config.get("entity_onnx_file", "onnx/model_int8.onnx"),
            score_threshold=float(config.get("entity_score_threshold", 0.6)),
            allowed_labels=set(labels),
            providers=providers,
            max_length=int(config.get("entity_max_length", 256)),
        )
    raise ValueError(f"unknown entity_extractor: {mode!r}")
