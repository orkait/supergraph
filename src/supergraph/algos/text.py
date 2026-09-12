"""Text pre-processing primitives."""

import re

__all__ = ["fts5_sanitize", "tokenize_unicode"]

_FTS5_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_FTS5_RESERVED_UPPER = {"AND", "OR", "NOT", "NEAR"}


def tokenize_unicode(text: str) -> list[str]:
    """Lowercase unicode word tokens, length > 1."""
    return [t.lower() for t in _FTS5_TOKEN_RE.findall(text or "") if len(t) > 1]


def fts5_sanitize(query: str) -> str:
    """Convert a natural-language query to a safe FTS5 MATCH expression.

    FTS5 treats ``?``, ``*``, ``:``, ``^``, ``+``, ``-``, ``~``, ``'``, ``"``,
    ``(``, ``)`` and uppercase ``AND``/``OR``/``NOT``/``NEAR`` as syntax, so
    raw questions like ``What is the user's favorite food?`` raise
    ``fts5: syntax error``.

    Extracts ``\\w+`` tokens, drops uppercase reserved keywords, lowercases,
    and joins with ``OR`` so BM25 ranks by ANY content word.
    """
    tokens = [
        t.lower()
        for t in _FTS5_TOKEN_RE.findall(query or "")
        if t not in _FTS5_RESERVED_UPPER
    ]
    return " OR ".join(tokens) if tokens else ""
