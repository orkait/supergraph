from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from superclaw.settings import LIMITS

IMAGE_TYPES = ("image/png", "image/bmp")
_PNGF_SCRIPT = 'set f to open for access POSIX file "{path}" with write permission\nwrite (the clipboard as «class PNGf») to f\nclose access f'


def _run(argv: list[str]) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(argv, capture_output=True, timeout=LIMITS.clipboard_timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _linux_image() -> tuple[bytes, str] | None:
    for mime in IMAGE_TYPES:
        for argv in (["xclip", "-selection", "clipboard", "-t", mime, "-o"], ["wl-paste", "--type", mime]):
            if not shutil.which(argv[0]):
                continue
            done = _run(argv)
            if done is not None and done.returncode == 0 and done.stdout:
                return done.stdout, mime
    return None


def _mac_image() -> tuple[bytes, str] | None:
    if shutil.which("pngpaste"):
        done = _run(["pngpaste", "-"])
        if done is not None and done.returncode == 0 and done.stdout:
            return done.stdout, "image/png"
    if shutil.which("osascript"):
        target = Path(tempfile.gettempdir()) / f"superclaw-clip-{os.getpid()}.png"
        done = _run(["osascript", "-e", _PNGF_SCRIPT.format(path=target)])
        try:
            if done is not None and done.returncode == 0 and target.is_file():
                return target.read_bytes(), "image/png"
        finally:
            target.unlink(missing_ok=True)
    return None


def image_bytes() -> tuple[bytes, str] | None:
    return _mac_image() if sys.platform == "darwin" else _linux_image()


def text() -> str:
    candidates = [["pbpaste"]] if sys.platform == "darwin" else [["xclip", "-selection", "clipboard", "-t", "text/plain", "-o"],
                                                                 ["wl-paste", "--type", "text/plain", "--no-newline"]]
    for argv in candidates:
        if not shutil.which(argv[0]):
            continue
        done = _run(argv)
        if done is not None and done.returncode == 0:
            return done.stdout.decode("utf-8", errors="replace")
    return ""


def parse_drop(pasted: str) -> list[Path]:
    stripped = pasted.strip()
    if not stripped or ("\n" in stripped.strip("\n") and stripped.count("\n") > LIMITS.drop_paths_max):
        return []
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        tokens = stripped.split()
    if not tokens or len(tokens) > LIMITS.drop_paths_max:
        return []
    found: list[Path] = []
    for token in tokens:
        raw = unquote(urlparse(token).path) if token.startswith("file://") else token
        path = Path(raw).expanduser()
        if not path.is_absolute() or not path.is_file():
            return []
        found.append(path)
    return found
