"""Entity extraction with ONNX TinyBERT NER + co-reference resolution."""
from __future__ import annotations

import hashlib
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_]+")


def slug(text: str) -> str:
    """Create a URL-safe slug. Appends a short hash suffix when truncated
    so distinct long names don't silently collide.
    """
    base = _SLUG_RE.sub("_", text.lower()).strip("_")
    if len(base) <= 40:
        return base
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:6]
    return base[:33].rstrip("_") + "_" + digest


_PRONOUN_MAP = {
    "she": "PER", "he": "PER", "him": "PER", "her": "PER",
    "they": "PER", "them": "PER",
    "it": "MISC", "this": "MISC", "that": "MISC",
}
_ALLOWED_LABELS = {"PER", "ORG", "LOC", "MISC"}


@dataclass
class Entity:
    text: str
    label: str
    score: float


class CoReferenceResolver:
    """Resolve pronouns to the most recently mentioned named entity."""

    def __init__(self):
        self._current_context: str | None = None

    def update_context(self, entity_name: str | None):
        if entity_name:
            self._current_context = entity_name

    def resolve(self, sentence: str) -> list[str]:
        """Return resolved named entity if pronouns found, else empty list."""
        if not self._current_context:
            return []
        words = re.findall(r'\b\w+\b', sentence.lower())
        for w in words:
            if w in _PRONOUN_MAP:
                return [self._current_context]
        return []


_extractors: dict[tuple, Any] = {}


def _get_extractor(model_dir: str | Path, max_length: int):
    # Key includes max_length so two callers with the same model but
    # different max_length don't share an extractor with the wrong
    # tokenizer truncation limit (bug #61). Pre-fix, only the path was
    # used, silently returning the first-initialized instance.
    key = (str(model_dir), int(max_length))
    if key not in _extractors:
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "ONNX entity extraction requires onnxruntime and tokenizers. "
                "Install with: pip install onnxruntime tokenizers"
            ) from e

        model_dir = Path(model_dir)
        tokenizer_path = model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            tokenizer_path = model_dir / "onnx" / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")

        onnx_path = model_dir / "onnx" / "model_int8.onnx"
        if not onnx_path.exists():
            onnx_path = model_dir / "onnx" / "model.onnx"
        if not onnx_path.exists():
            onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found in {model_dir}")

        config_path = model_dir / "config.json"
        if not config_path.exists():
            config_path = model_dir / "onnx" / "config.json"
        id2label = {}
        if config_path.exists():
            cfg = json.loads(config_path.read_text())
            raw_id2label = cfg.get("id2label", {})
            if raw_id2label and "LABEL_0" in raw_id2label.values():
                id2label = {
                    "0": "O",
                    "1": "B-PER", "2": "I-PER",
                    "3": "B-ORG", "4": "I-ORG",
                    "5": "B-LOC", "6": "I-LOC",
                    "7": "B-MISC", "8": "I-MISC",
                }
            else:
                id2label = {str(k): str(v) for k, v in raw_id2label.items()}

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_truncation(max_length=max_length)

        from supergraph.core.compute_profile import get_profile
        profile = get_profile()
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if profile.has_gpu:
            providers = [profile.gpu_provider, "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            sess_options.intra_op_num_threads = profile.ner_threads
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
        print(f"  [NER] Model loaded. Provider: {session.get_providers()[0]} threads={profile.ner_threads}")
        input_names = {i.name for i in session.get_inputs()}

        _extractors[key] = {
            "tokenizer": tokenizer,
            "session": session,
            "id2label": id2label,
            "input_names": input_names,
        }
    return _extractors[key]


def _decode_entities(text: str, offsets: list[tuple[int, int]],
                     labels: list[str], scores: np.ndarray,
                     score_threshold: float) -> list[Entity]:
    """Decode token-level BIO labels into entity spans."""
    out: list[Entity] = []
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
        entity_text = text[current_start:current_end].strip()
        if entity_text and len(entity_text) >= 3 and avg_score >= score_threshold:
            out.append(Entity(text=entity_text, label=current_type, score=avg_score))
        current_start = current_end = None
        current_type = None
        current_scores = []

    for (start, end), label, score in zip(offsets, labels, scores):
        if start == end or label == "O":
            flush()
            continue
        if "-" in label:
            prefix, ent_type = label.split("-", 1)
        else:
            prefix, ent_type = "B", label
        if ent_type not in _ALLOWED_LABELS:
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
    return out


def extract_batch(texts: list[str], model_dir: str | Path | None = None,
                  score_threshold: float = 0.6,
                  max_length: int = 256) -> list[list[Entity]]:
    """Extract named entities from multiple texts using ONNX TinyBERT NER."""
    if not texts:
        return []
    if model_dir is None:
        return [[] for _ in texts]
    
    extractor = _get_extractor(model_dir, max_length)
    encodings = [extractor["tokenizer"].encode(t) for t in texts]
    
    # Simple padding
    max_len = max(len(e.ids) for e in encodings)
    input_ids = []
    attention_mask = []
    token_type_ids = []
    
    for e in encodings:
        pad_len = max_len - len(e.ids)
        input_ids.append(e.ids + [0] * pad_len)
        attention_mask.append(e.attention_mask + [0] * pad_len)
        if "token_type_ids" in extractor["input_names"]:
            token_type_ids.append([0] * max_len)

    feed = {
        "input_ids": np.array(input_ids, dtype=np.int64),
        "attention_mask": np.array(attention_mask, dtype=np.int64),
    }
    if "token_type_ids" in extractor["input_names"]:
        feed["token_type_ids"] = np.array(token_type_ids, dtype=np.int64)

    all_logits = extractor["session"].run(None, feed)[0]
    
    results = []
    for i, logits in enumerate(all_logits):
        # Softmax
        x = logits[:len(encodings[i].ids)] - np.max(logits[:len(encodings[i].ids)], axis=-1, keepdims=True)
        exp_x = np.exp(x)
        probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        pred_ids = np.argmax(probs, axis=-1)
        id2label = extractor["id2label"]
        labels = [id2label.get(str(int(idx)), "O") for idx in pred_ids]
        scores = np.max(probs, axis=-1)
        results.append(_decode_entities(texts[i], list(encodings[i].offsets), labels, scores, score_threshold))
        
    return results


def extract_entities(text: str, model_dir: str | Path | None = None,
                     score_threshold: float = 0.6,
                     max_length: int = 256) -> list[Entity]:
    """Extract named entities from text using ONNX TinyBERT NER.

    Returns list of Entity dataclasses sorted by position in text.
    Returns empty list if text is empty or no model_dir provided.
    """
    res = extract_batch([text], model_dir, score_threshold, max_length)
    return res[0] if res else []
