from __future__ import annotations

import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from superclaw.runtime import clip, compact
from superclaw.settings import AUDIO_FORMAT, LIMITS, YTDLP_BIN
from superclaw.tools import PathEscapes, Permission, Result, Safety, SideEffect, Tool, ToolContext, jail, relative
from superclaw.tools.budget import Category
from superclaw.tools.fetch import HEADERS, Unsafe, open_url, validate

KINDS = ("auto", "video", "audio", "file")
MEDIA_KINDS = ("video", "audio")
UNSUPPORTED = "Unsupported URL"
OUTPUT_TEMPLATE = "%(title).80s-%(id)s.%(ext)s"
FALLBACK_NAME = "download"
_FILENAME = re.compile(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', re.IGNORECASE)
which = shutil.which
run = subprocess.run


class Skipped(RuntimeError):
    pass


def media_args(tool: str, url: str, dest: Path, kind: str) -> list[str]:
    argv = [tool, "--no-playlist", "--restrict-filenames", "--no-progress", "--quiet", "--no-warnings", "--no-overwrites",
            "--max-filesize", str(LIMITS.download_bytes_max), "--paths", str(dest), "--output", OUTPUT_TEMPLATE, "--print", "after_move:filepath"]
    if kind == "audio":
        argv += ["--extract-audio", "--audio-format", AUDIO_FORMAT]
    return [*argv, url]


def media(tool: str, url: str, dest: Path, kind: str) -> list[Path]:
    proc = run(media_args(tool, url, dest, kind), capture_output=True, text=True, timeout=LIMITS.download_timeout_s)
    paths = [Path(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode == 0 and paths:
        return paths
    if kind == "auto" and UNSUPPORTED in proc.stderr:
        raise Skipped(UNSUPPORTED)
    raise RuntimeError(clip(proc.stderr.strip() or "yt-dlp saved nothing", LIMITS.preview_error_chars))


def filename_of(response: Any, url: str) -> str:
    match = _FILENAME.search(str(response.headers.get("Content-Disposition") or ""))
    raw = urllib.parse.unquote(match.group(1)) if match else Path(urllib.parse.urlparse(url).path).name
    return Path(raw.strip()).name or FALLBACK_NAME


def plain(url: str, dest: Path, name: str) -> Path:
    parsed = validate(url)
    with open_url(urllib.request.Request(parsed.geturl(), headers=HEADERS)) as response:
        target = dest / (Path(name).name if name else filename_of(response, response.geturl()))
        if target.exists():
            raise FileExistsError(f"{target.name} already exists in {dest}")
        size = 0
        with target.open("wb") as out:
            while chunk := response.read(LIMITS.download_chunk_bytes):
                size += len(chunk)
                if size > LIMITS.download_bytes_max:
                    out.close()
                    target.unlink()
                    raise ValueError(f"larger than {compact(LIMITS.download_bytes_max)}B")
                out.write(chunk)
    return target


class Download(Tool):
    name = "download"
    deferred = True
    description = (
        "Download a URL into the workspace. Video and audio pages (YouTube and every site yt-dlp knows) go through yt-dlp when it is installed, "
        "kind=audio extracts audio; anything else, or kind=file, is saved as a plain file. Returns the saved paths and sizes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Public http or https URL."},
            "kind": {"type": "string", "enum": list(KINDS), "description": "auto tries yt-dlp then a plain download; video and audio require yt-dlp; file skips it.", "default": "auto"},
            "dest": {"type": "string", "description": "Directory inside the workspace to save into.", "default": "."},
            "name": {"type": "string", "description": "File name for a plain download; defaults to the server's name."},
        },
        "required": ["url"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.NETWORK, Permission.PROMPT, "Downloads a model-chosen URL from this host into the workspace.")
    output_category = Category.PROCESS

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        url = str(args.get("url") or "").strip()
        kind = str(args.get("kind") or "auto")
        if not url:
            return Result.error("Error: url must not be empty")
        if kind not in KINDS:
            return Result.error(f"Error: kind must be one of {', '.join(KINDS)}")
        try:
            dest = jail(ctx.roots, str(args.get("dest") or "."))
            validate(url)
            dest.mkdir(parents=True, exist_ok=True)
            saved = self.save(url, dest, kind, str(args.get("name") or ""))
        except (PathEscapes, Unsafe, FileExistsError, ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as e:
            return Result.error(f"Error: download failed: {e}")
        lines = [f"Saved {relative(ctx.roots, path)} ({compact(path.stat().st_size)}B)" for path in saved]
        return Result.success("\n".join(lines), changed_files=[relative(ctx.roots, path) for path in saved])

    @staticmethod
    def save(url: str, dest: Path, kind: str, name: str) -> list[Path]:
        tool = which(YTDLP_BIN) if kind != "file" else None
        if tool:
            try:
                return media(tool, url, dest, kind)
            except Skipped:
                pass
        elif kind in MEDIA_KINDS:
            raise RuntimeError(f"{YTDLP_BIN} is not installed; uv tool install {YTDLP_BIN}")
        return [plain(url, dest, name)]
