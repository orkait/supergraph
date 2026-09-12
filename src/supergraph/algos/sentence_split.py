"""Sentence splitting primitives for query-time expansion.

Lightweight regex-based splitter with guards for common false boundaries:
abbreviations, URLs, decimal numbers, and ellipsis.

No external dependencies (nltk, spaCy) - keeps supergraph zero-infra.
"""

import re

__all__ = ["split_sentences"]

_ABBREV_RE = re.compile(
    r"(\b(?:mr|mrs|ms|dr|prof|sr|jr|st|ave|blvd|dept|vol|vs|etc|"
    r"inc|ltd|co|corp|gov|gen|col|sgt|capt|maj|lt|pvt|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"mon|tue|wed|thu|fri|sat|sun|"
    r"e\.g|i\.e|cf|al|approx|no|fig|eq|ref|sec|chap|app))\.",
    re.IGNORECASE,
)

# Match a capital-letter initial ("A.") preceded by start-of-string or
# whitespace and followed by whitespace + another capital. The first group
# captures any leading whitespace so the substitution can reinstate it
# alongside the preserved letter. Works for single ("A. Smith") and chained
# ("A. B. Smith") initials — chained cases iterate left-to-right and each
# period is replaced in turn, preserving the whitespace between them.
_INITIALS_RE = re.compile(r"(^|\s)([A-Z])\.(?=\s+[A-Z])")

_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")

_PLACEHOLDER_ABB = "\x00ABB\x00"
_PLACEHOLDER_INIT = "\x00INIT\x00"


def split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Returns a list of non-empty, stripped sentence strings.
    Returns ``[text]`` for short text (< 20 chars) or if no boundary found.
    Returns ``[]`` for empty/whitespace-only input.

    Guards against false splits on:
        - Common abbreviations: Dr., Mr., e.g., i.e., etc.
        - Initials: A. B. Smith
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    if len(text) < 20:
        return [text]

    return _split(text)


def _split(text: str) -> list[str]:
    # Replace only the trailing period of abbreviations, preserving the word
    protected = _ABBREV_RE.sub(r"\1" + _PLACEHOLDER_ABB, text)

    # Replace periods between initials, preserving the letter. The lookahead
    # consumes no characters after the period, so the substitution only
    # needs to reinstate the leading whitespace (group 1) + letter (group 2)
    # + placeholder. Original whitespace between initials is untouched,
    # which matters for chained forms like "A. B. Smith".
    protected = _INITIALS_RE.sub(lambda m: m.group(1) + m.group(2) + _PLACEHOLDER_INIT, protected)

    parts = _BOUNDARY_RE.split(protected)

    sentences: list[str] = []
    for part in parts:
        part = part.replace(_PLACEHOLDER_ABB, ".").replace(_PLACEHOLDER_INIT, ".")
        part = part.strip()
        if part:
            sentences.append(part)

    if len(sentences) <= 1:
        return [text] if text else []

    return sentences
