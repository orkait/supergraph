
from __future__ import annotations

import logging
import re
import string

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


def llm_call(prompt: str, max_tokens: int = 1000, temperature: float = 0.0, _retries: int | None = None) -> str:
    from supergraph.llm_runner import get_shared_runner
    return get_shared_runner().call_sync(prompt, max_tokens=max_tokens, temperature=temperature)


def health_check() -> bool:
    result = llm_call("Say OK", max_tokens=500)
    if not result:
        raise RuntimeError(
            "LLM health check failed: got empty response. "
            "Check tools/autoresearch/config.json that active_provider + "
            "provider_fallback_order resolve to at least one reachable model."
        )
    return True


def _normalize_answer(s: str) -> str:
    s = s.replace(',', '')
    s = re.sub(r'\b(a|an|the|and)\b', ' ', s.lower())
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())


def _f1_score(prediction: str, gold: str) -> float:
    from collections import Counter
    try:
        from nltk.stem import PorterStemmer
        _stemmer = PorterStemmer()
        stem = _stemmer.stem
    except ImportError:
        stem = lambda w: w

    pred_tokens = [stem(w) for w in _normalize_answer(prediction).split()]
    gold_tokens = [stem(w) for w in _normalize_answer(gold).split()]
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_f1(prediction: str, gold: str, category: int | None = None) -> float:
    if category == 5:
        low = prediction.lower()
        if 'no information available' in low or 'not mentioned' in low:
            return 1.0
        return 0.0

    if category == 3:
        gold = gold.split(';')[0].strip()

    if category == 1:
        import numpy as np
        preds = [p.strip() for p in prediction.split(',')]
        golds = [g.strip() for g in gold.split(',')]
        return float(np.mean([max([_f1_score(p, g) for p in preds]) for g in golds]))

    return _f1_score(prediction, gold)


_JUDGE_PROMPT = """\
You are grading a short-answer question.

Question: {question}
Gold answer: {gold}
Predicted answer: {prediction}

Mark the predicted answer CORRECT if it is semantically equivalent to the
gold answer OR captures the same essential fact, even with different
wording, abbreviation, or paraphrase. Mark it INCORRECT only if it
contradicts the gold answer, omits the essential fact, or fabricates
unsupported content.

For adversarial questions (gold answer is "no information available" or
similar), CORRECT requires the prediction to also acknowledge that no
information is available. Any confidently-stated answer is INCORRECT.

Respond with a single word: CORRECT or INCORRECT.
"""


def compute_llm_judge(
    prediction: str,
    gold: str,
    question: str,
    category: int | None = None,
    debug: bool = False,
) -> float:
    pred = (prediction or "").strip()
    if not pred:
        return 0.0
    if category == 5:
        low = pred.lower()
        if any(phrase in low for phrase in (
            "no information available", "not mentioned", "cannot answer",
            "don't know", "do not know", "not enough information",
        )):
            return 1.0
        return 0.0

    prompt = _JUDGE_PROMPT.format(
        question=(question or "").strip(),
        gold=(gold or "").strip(),
        prediction=pred,
    )
    verdict = llm_call(prompt, max_tokens=2000, temperature=0.0)
    if debug:
        print(f"  [judge] raw verdict: {verdict!r}")
    if not verdict:
        return 0.0
    upper = verdict.upper()
    last_incorrect = upper.rfind("INCORRECT")
    last_correct = upper.rfind("CORRECT")
    if last_incorrect == -1 and last_correct == -1:
        return 0.0
    if last_incorrect != -1 and (last_correct == -1 or last_correct <= last_incorrect + 2):
        return 0.0
    return 1.0
