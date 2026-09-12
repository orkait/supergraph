from pathlib import Path
from supergraph.ingest.base import Ingestor


_BUILTIN_EXT_MAP: dict[str, str] = {
    "txt": "markitdown", "md": "markitdown", "html": "markitdown", "htm": "markitdown",
    "csv": "markitdown", "json": "markitdown", "xml": "markitdown",
    "docx": "markitdown", "pptx": "markitdown", "xlsx": "markitdown", "zip": "markitdown",
    "pdf": "pymupdf4llm",
    "png": "markitdown", "jpg": "markitdown", "jpeg": "markitdown",
    "gif": "markitdown", "webp": "markitdown",
    "wav": "audio", "mp3": "audio", "ogg": "audio", "flac": "audio",
}

_DOCLING_EXCLUSIVE: dict[str, str] = {
    "tex": "docling", "adoc": "docling",
    "tif": "docling", "tiff": "docling", "bmp": "docling",
    "m4a": "docling", "aac": "docling",
    "mp4": "docling", "avi": "docling", "mov": "docling",
}


def _build_builtin_ext_map() -> dict[str, str]:
    ext_map = dict(_BUILTIN_EXT_MAP)
    try:
        import docling as _  # noqa
        ext_map.update(_DOCLING_EXCLUSIVE)
    except ImportError:
        pass
    return ext_map


def _make_builtin_ingestor(name: str) -> Ingestor:
    if name == "markitdown":
        from supergraph.ingest.markitdown_ingestor import MarkItDownIngestor
        return MarkItDownIngestor()
    if name == "pymupdf4llm":
        from supergraph.ingest.pymupdf4llm_ingestor import PyMuPDF4LLMIngestor
        return PyMuPDF4LLMIngestor()
    if name == "docling":
        from supergraph.ingest.docling_ingestor import DoclingIngestor
        return DoclingIngestor()
    if name == "audio":
        raise ValueError(
            "Audio ingestion requires docling with the ASR extra. "
            "Install with: pip install 'supergraphdb[ingest]' 'docling[asr]'"
        )
    raise ValueError(f"Unknown built-in ingestor: {name!r}")


class IngestorRegistry:

    def __init__(self) -> None:
        self._ext_map: dict[str, str] = _build_builtin_ext_map()
        self._instances: dict[str, Ingestor] = {}

    def register(self, ingestor: Ingestor) -> None:
        self._instances[ingestor.name] = ingestor
        for ext in ingestor.supported_extensions:
            self._ext_map[ext] = ingestor.name

    def resolve(self, file_path: str, using: str | None = None) -> Ingestor:
        if using:
            name = using
        else:
            ext = Path(file_path).suffix.lstrip(".").lower()
            if ext not in self._ext_map:
                supported = sorted(set(self._ext_map.keys()))
                raise ValueError(
                    f"Unsupported format: .{ext}. Supported: {supported}"
                )
            name = self._ext_map[ext]

        if name not in self._instances:
            self._instances[name] = _make_builtin_ingestor(name)
        return self._instances[name]

    def list(self) -> list[dict]:
        seen: dict[str, list[str]] = {}
        for ext, name in self._ext_map.items():
            seen.setdefault(name, []).append(ext)
        result = []
        for name, exts in seen.items():
            entry: dict = {"name": name, "formats": sorted(exts)}
            result.append(entry)
        return result
