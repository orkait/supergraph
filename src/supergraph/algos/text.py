
import re

__all__ = ["fts5_sanitize", "tokenize_unicode"]

_FTS5_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_FTS5_RESERVED_UPPER = {"AND", "OR", "NOT", "NEAR"}


def tokenize_unicode(text: str) -> list[str]:
    return [t.lower() for t in _FTS5_TOKEN_RE.findall(text or "") if len(t) > 1]


def fts5_sanitize(query: str) -> str:
    tokens = [
        t.lower()
        for t in _FTS5_TOKEN_RE.findall(query or "")
        if t not in _FTS5_RESERVED_UPPER
    ]
    return " OR ".join(tokens) if tokens else ""
