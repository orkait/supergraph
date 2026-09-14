from __future__ import annotations

from typing import Any

from supergraph.core.errors import SuperGraphError

from superclaw.documents import Documents
from superclaw.runtime import clip, count
from superclaw.settings import LIMITS
from superclaw.tools import Permission, Result, Safety, SideEffect, Tool, ToolContext, jail, relative


def describe(doc: Any, shown: str, pinned: bool, ttl_days: int) -> str:
    keep = "pinned" if pinned else f"expires in {ttl_days} days"
    return (f"Ingested {shown} as {doc.id}: {count(doc.chunks, 'chunk')} via {doc.parser or 'direct'}, confidence {doc.confidence:.2f}, {keep}; "
            f"recall with a query finds its chunks, recall doc={doc.id} reads them in order")


class Ingest(Tool):
    name = "ingest"
    deferred = True
    description = (
        "Parse a workspace file into the brain: pdf, docx, pptx, xlsx, html, csv, json, markdown, text, images (and audio with the audio extra) become "
        "searchable chunks that recall finds by query. Kept for a month unless pin is true. Use it for a document worth keeping, not for every file you read."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace path of the file."},
            "ttl_days": {"type": "integer", "minimum": 1, "description": "Days to keep the chunks.", "default": LIMITS.ingest_ttl_days},
            "pin": {"type": "boolean", "description": "Keep it until retracted instead of expiring.", "default": False},
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.WRITE, Permission.PROMPT, "Parses a file into the brain's chunk index, kept for a month unless pinned.")

    def __init__(self, documents: Documents | None = None) -> None:
        self._documents = documents

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        if self._documents is None:
            return Result.error("Error: no document store in this run")
        target = jail(ctx.roots, str(args.get("path") or ""))
        if not target.is_file():
            return Result.error(f"Error: {args.get('path')} is not a file")
        ttl_days = int(args.get("ttl_days") or LIMITS.ingest_ttl_days)
        pinned = bool(args.get("pin"))
        try:
            doc = self._documents.ingest(target, session_id=ctx.session_id, ttl_days=ttl_days, pin=pinned)
        except (ImportError, ValueError, SuperGraphError) as e:
            return Result.error(f"Error: {clip(str(e), LIMITS.preview_error_chars)}")
        return Result.success(describe(doc, relative(ctx.roots, target), pinned, ttl_days))
