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

_DOCLING_EXCLUSIVE = {
    "tex": "docling",
    "adoc": "docling",
    "tif": "docling",
    "tiff": "docling",
    "bmp": "docling",
}

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
    _docling_formats = ["pdf", "docx", "pptx", "xlsx", "md", "html", "csv",
                        "png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp",
                        "tex", "adoc"]
    import importlib.metadata as _im
    import importlib.util as _il

    def _installed(dist: str) -> bool:
        try:
            _im.distribution(dist)
            return True
        except _im.PackageNotFoundError:
            return False

    docling_available = _il.find_spec("docling") is not None
    vision_available = _installed("llama-cpp-python")
    stt_available = _il.find_spec("faster_whisper") is not None

    return [
        {"name": "markitdown", "formats": ["txt", "md", "html", "csv", "json", "xml", "docx", "pptx", "xlsx", "pdf", "zip", "png", "jpg"], "tier": 1, "extra": "ingest"},
        {"name": "pymupdf4llm", "formats": ["pdf"], "tier": 2, "extra": "ingest"},
        {"name": "docling", "formats": _docling_formats, "tier": 3, "available": docling_available, "extra": "ingest-pro"},
        {"name": "vision", "formats": ["png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff"], "tier": 4, "available": vision_available, "extra": "vision"},
        {"name": "whisper", "formats": ["wav", "mp3", "ogg", "flac", "m4a", "opus", "webm"], "tier": 4, "available": stt_available, "extra": "audio"},
    ]
