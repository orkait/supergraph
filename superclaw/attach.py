from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path

from superclaw.settings import LIMITS
from superclaw.tools import jail
from superclaw.tools.budget import Category, budget_output

IMAGE_PREFIX = "image/"
TEXT_PREFIX = "text/"
TEXT_TYPES = ("application/json", "application/xml", "application/x-yaml", "application/toml")


@dataclass
class Attachments:
    text: str = ""
    images: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _kind(path: Path) -> str:
    guess = mimetypes.guess_type(path.name)[0] or ""
    if guess.startswith(IMAGE_PREFIX):
        return "image"
    if guess.startswith(TEXT_PREFIX) or guess in TEXT_TYPES or not guess:
        return "text"
    return guess


def _block(label: str, body: str) -> str:
    return f'<attachment path="{label}">\n{body}\n</attachment>'


def read(paths: list[str], roots: tuple[Path, ...]) -> Attachments:
    found = Attachments()
    blocks: list[str] = []
    for raw in paths:
        try:
            target = jail(roots, raw)
        except ValueError as e:
            found.problems.append(f"{raw}: {e}")
            continue
        if not target.is_file():
            found.problems.append(f"{raw}: not a file")
            continue
        data = target.read_bytes()
        kind = _kind(target)
        if kind == "image":
            if len(data) > LIMITS.attachment_image_bytes:
                found.problems.append(f"{raw}: image is larger than {LIMITS.attachment_image_bytes} bytes")
                continue
            mime = mimetypes.guess_type(target.name)[0] or "image/png"
            found.images.append(f"data:{mime};base64,{base64.b64encode(data).decode()}")
            blocks.append(_block(raw, "(attached as an image)"))
            continue
        if kind != "text":
            found.problems.append(f"{raw}: {kind} is not text or an image")
            continue
        body = data[: LIMITS.read_file_bytes].decode("utf-8", errors="replace")
        blocks.append(_block(raw, budget_output(body, Category.FILE).text))
    found.text = "\n\n".join(blocks)
    return found
