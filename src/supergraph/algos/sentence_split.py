
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

_INITIALS_RE = re.compile(r"(^|\s)([A-Z])\.(?=\s+[A-Z])")

_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")

_PLACEHOLDER_ABB = "\x00ABB\x00"
_PLACEHOLDER_INIT = "\x00INIT\x00"


def split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []

    text = text.strip()

    if len(text) < 20:
        return [text]

    return _split(text)


def _split(text: str) -> list[str]:
    protected = _ABBREV_RE.sub(r"\1" + _PLACEHOLDER_ABB, text)

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
