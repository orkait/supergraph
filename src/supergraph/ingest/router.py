"""Tiered ingestor routing: deterministic parsers first, VLM as fallback."""
from pathlib import Path
from supergraph.ingest.base import IngestResult

_PLAINTEXT_EXTS = {"txt", "md"}

EXTENSION_MAP = {
    "txt": "markitdown", "md": "markitdown", "html": "markitdown", "htm": "markitdown",
    "csv": "markitdown", "json": "markitdown", "xml": "markitdown",
    "docx": "markitdown", "pptx": "markitdown", "xlsx": "markitdown",
    "zip": "markitdown",
    "pdf": "pymupdf4llm",
    "png": "markitdown", "jpg": "markitdown", "jpeg": "markitdown",
    "gif": "markitdown", "webp": "markitdown",
}

# Formats only docling can handle - added when docling is installed
_DOCLING_EXCLUSIVE = {
    "tex": "docling",           # LaTeX
    "adoc": "docling",          # AsciiDoc
    "tif": "docling",           # TIFF images
    "tiff": "docling",
    "bmp": "docling",           # BMP images
}

# Audio formats routed through the whisper ingestor when the `[audio]` extra
# is installed. Added dynamically so users without the extra still get a
# clear "Unsupported format" error (router default) rather than an
# obscure faster-whisper ImportError at convert time.
_AUDIO_EXTS = {
    "wav": "whisper",
    "mp3": "whisper",
    "ogg": "whisper",
    "flac": "whisper",
    "m4a": "whisper",
    "opus": "whisper",
    "webm": "whisper",
}

try:
    import docling as _docling_check  # noqa
    EXTENSION_MAP.update(_DOCLING_EXCLUSIVE)
except ImportError:
    pass

try:
    import faster_whisper as _fw_check  # noqa
    EXTENSION_MAP.update(_AUDIO_EXTS)
except ImportError:
    pass

SUPPORTED_EXTENSIONS = set(EXTENSION_MAP.keys())

_ingestor_cache = {}


def _kwargs_cache_key(kwargs: dict) -> tuple:
    """Produce a hashable, order-independent key from kwargs.

    Used to distinguish cached ingestors with different configuration
    (e.g., max_tokens=500 vs max_tokens=1000). Pre-fix, the cache keyed
    by ingestor name only, so a second construction with different
    kwargs silently returned the first instance with the ORIGINAL kwargs
    still active (bug #59). Values must be hashable; non-hashable values
    fall back to their repr.
    """
    items = []
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        try:
            hash(v)
            items.append((k, v))
        except TypeError:
            items.append((k, repr(v)))
    return tuple(items)


def _get_ingestor(name: str, **kwargs):
    cache_key = (name, _kwargs_cache_key(kwargs))
    if cache_key not in _ingestor_cache:
        if name == "markitdown":
            from supergraph.ingest.markitdown_ingestor import MarkItDownIngestor
            _ingestor_cache[cache_key] = MarkItDownIngestor(**kwargs)
        elif name == "pymupdf4llm":
            from supergraph.ingest.pymupdf4llm_ingestor import PyMuPDF4LLMIngestor
            _ingestor_cache[cache_key] = PyMuPDF4LLMIngestor()
        elif name == "docling":
            from supergraph.ingest.docling_ingestor import DoclingIngestor
            _ingestor_cache[cache_key] = DoclingIngestor()
        elif name == "whisper":
            from supergraph.ingest.whisper_ingestor import WhisperIngestor
            _ingestor_cache[cache_key] = WhisperIngestor()
        else:
            raise ValueError(f"Unknown ingestor: {name!r}. Available: markitdown, pymupdf4llm, docling, whisper")
    return _ingestor_cache[cache_key]


def select_ingestor(file_path: str, using: str | None = None) -> str:
    if using:
        return using
    ext = Path(file_path).suffix.lstrip(".").lower()
    if ext not in EXTENSION_MAP:
        raise ValueError(f"Unsupported format: .{ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    return EXTENSION_MAP[ext]


def ingest_file(file_path: str, using: str | None = None, **kwargs) -> IngestResult:
    ext = Path(file_path).suffix.lstrip(".").lower()
    if ext in _PLAINTEXT_EXTS and using is None:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        return IngestResult(markdown=text, parser_used="direct", confidence=1.0,
                           metadata={"source": file_path})
    name = select_ingestor(file_path, using)
    ingestor = _get_ingestor(name, **kwargs)
    return ingestor.convert(file_path)


def list_ingestors() -> list[dict]:
    """Report available ingestors + their registered extensions.

    Tier stack:
      1. markitdown   (always, part of [ingest])
      2. pymupdf4llm  (PDF-only, part of [ingest])
      3. docling      (heavier PDF + OCR + LaTeX/AsciiDoc, [ingest-pro])
      4. vision       (image captioning via VLM sidecar, [vision])
      4. whisper      (speech-to-text via faster-whisper, [audio])

    Tiers 4 are modality-specific fallbacks, not a linear escalation.
    """
    _docling_formats = ["pdf", "docx", "pptx", "xlsx", "md", "html", "csv",
                        "png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp",
                        "tex", "adoc"]
    # Probe by importlib metadata, not by import. The vision sidecar's
    # llama-cpp-python loads its native lib at module import; on hosts with
    # missing/broken CUDA libs that import raises RuntimeError (not
    # ImportError) and crashes the whole capability registry. Distribution
    # metadata is a pure file-system read with no side effects.
    import importlib.metadata as _im
    import importlib.util as _il

    def _installed(dist: str) -> bool:
        try:
            _im.distribution(dist)
            return True
        except _im.PackageNotFoundError:
            return False

    docling_available = _il.find_spec("docling") is not None
    # Check parent dist; find_spec("llama_cpp.server") would import
    # llama_cpp.__init__ which triggers the dlopen we're avoiding.
    vision_available = _installed("llama-cpp-python")
    stt_available = _il.find_spec("faster_whisper") is not None

    return [
        {"name": "markitdown", "formats": ["txt", "md", "html", "csv", "json", "xml", "docx", "pptx", "xlsx", "pdf", "zip", "png", "jpg"], "tier": 1, "extra": "ingest"},
        {"name": "pymupdf4llm", "formats": ["pdf"], "tier": 2, "extra": "ingest"},
        {"name": "docling", "formats": _docling_formats, "tier": 3, "available": docling_available, "extra": "ingest-pro"},
        {"name": "vision", "formats": ["png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff"], "tier": 4, "available": vision_available, "extra": "vision"},
        {"name": "whisper", "formats": ["wav", "mp3", "ogg", "flac", "m4a", "opus", "webm"], "tier": 4, "available": stt_available, "extra": "audio"},
    ]
