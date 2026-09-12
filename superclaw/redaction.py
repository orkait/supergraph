from __future__ import annotations

import re

REDACTED = "[REDACTED]"

_PATTERNS = [
    re.compile(r"\bsk-ant-(?:api\d{2}-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{12,}"),
    re.compile(r"\bAIza[0-9A-Za-z\-_]{35,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
]
_HEADER = re.compile(r"(?i)\b(authorization|proxy-authorization|x-api-key|api-key|cookie|set-cookie)(\s*:\s*)([^\r\n]+)")
_ASSIGN = re.compile(r"(?i)\b([A-Za-z_][A-Za-z0-9_.-]*?(?:password|passwd|passphrase|secret|api_?key|private_key|token))(\s*[=:]\s*)(\"[^\"]*\"|'[^']*'|[^\s&,;]+)")


def redact(text: str) -> tuple[str, bool]:
    out = text
    for pattern in _PATTERNS:
        out = pattern.sub(REDACTED, out)
    out = _HEADER.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", out)
    out = _ASSIGN.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", out)
    return out, out != text
