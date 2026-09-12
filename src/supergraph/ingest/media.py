"""Cloud multimodal understanding: media bytes -> text, for ingestion.

supergraph OWNS understanding media. It is an ingestion function - media in,
understood text out, then stored + embedded like any DOCUMENT - the same shape
as NL->DSL. A caller (e.g. a harness) only fetches the bytes; the understanding
lives here.

Modalities: image / audio / video go through a cloud multimodal model (the model
decodes the bytes - no local codecs needed). PDF is handled locally (pymupdf text
extraction, with a render->vision fallback for scanned pages).
"""
from __future__ import annotations

import base64

from supergraph.ingest.llm.resolve import build_provider_chain
from supergraph.llm_runner import LLMRunner

# per-modality model defaults (the model must accept that input modality)
DEFAULT_VISION_MODELS = ["openrouter/openai/gpt-4o-mini", "openrouter/google/gemini-3.5-flash"]
DEFAULT_AUDIO_MODELS = ["openrouter/google/gemini-3.5-flash", "openrouter/openai/gpt-audio-mini"]
DEFAULT_VIDEO_MODELS = ["openrouter/google/gemini-3.5-flash"]

_IMAGE_PROMPT = (
    "Describe this image in detail - objects, any visible text, people, setting, "
    "colors, and anything notable. Be factual and concise."
)
_AUDIO_PROMPT = "Transcribe the speech and describe any other sounds in this audio. Be concise."
_VIDEO_PROMPT = (
    "Describe this video - what happens, the setting, any people/objects, on-screen "
    "text, and notable audio. Be factual and concise."
)
_PDF_VISION_PROMPT = "Read this document page image and return its text and a brief description."

# mime audio subtype -> the `format` the model expects
_AUDIO_FMT = {"mpeg": "mp3", "mp3": "mp3", "x-mp3": "mp3", "wav": "wav", "x-wav": "wav",
              "ogg": "ogg", "mp4": "mp4", "m4a": "mp4", "flac": "flac", "webm": "webm"}
_PDF_TEXT_MIN = 16
_PDF_MAX_PAGES = 5


class MediaUnsupported(ValueError):
    """Raised when a mime type can't be understood."""


def _run(messages: list[dict], default_models: list[str], models: list[str] | None,
         max_tokens: int) -> str:
    chain = build_provider_chain(models or default_models)
    if not chain:
        raise RuntimeError(
            "no cloud providers resolved; set a provider key "
            "(e.g. OPENROUTER_API_KEY / GOOGLE_AISTUDIO_API_KEY)"
        )
    return (LLMRunner(chain).complete_messages(messages, max_tokens=max_tokens, temperature=0.0) or "").strip()


def _build(data: bytes, mime: str, prompt: str | None) -> tuple[list[dict], list[str]]:
    b64 = base64.b64encode(data).decode()
    if mime.startswith("image/"):
        return ([{"role": "user", "content": [
            {"type": "text", "text": prompt or _IMAGE_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}], DEFAULT_VISION_MODELS)
    if mime.startswith("audio/"):
        fmt = _AUDIO_FMT.get(mime.split("/", 1)[1], mime.split("/", 1)[1] or "mp3")
        return ([{"role": "user", "content": [
            {"type": "text", "text": prompt or _AUDIO_PROMPT},
            {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
        ]}], DEFAULT_AUDIO_MODELS)
    if mime.startswith("video/"):
        return ([{"role": "user", "content": [
            {"type": "text", "text": prompt or _VIDEO_PROMPT},
            {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}], DEFAULT_VIDEO_MODELS)
    raise MediaUnsupported(f"no cloud understanding path for mime {mime!r}")


def _fitz():
    try:
        import fitz  # pymupdf
    except ImportError as e:
        raise MediaUnsupported(
            "PDF understanding needs pymupdf: install supergraph[ingest]"
        ) from e
    return fitz


def _pdf_text(data: bytes) -> str:
    fitz = _fitz()
    with fitz.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc).strip()


def _pdf_page_pngs(data: bytes, max_pages: int) -> list[bytes]:
    fitz = _fitz()
    out: list[bytes] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            out.append(page.get_pixmap(dpi=120).tobytes("png"))
    return out


def _understand_pdf(data: bytes, *, models: list[str] | None, prompt: str | None,
                    max_tokens: int) -> str:
    text = _pdf_text(data)
    if len(text) >= _PDF_TEXT_MIN:
        return text[:8000]
    # scanned / image-only PDF: render pages and read them with vision
    pages = _pdf_page_pngs(data, _PDF_MAX_PAGES)
    if not pages:
        raise MediaUnsupported("pdf has no extractable text and no renderable pages")
    content: list[dict] = [{"type": "text", "text": prompt or _PDF_VISION_PROMPT}]
    for png in pages:
        b64 = base64.b64encode(png).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return _run([{"role": "user", "content": content}], DEFAULT_VISION_MODELS, models, max_tokens)


def understand_media(
    data: bytes, mime: str, *, models: list[str] | None = None,
    prompt: str | None = None, max_tokens: int = 512,
) -> str:
    """Understand media bytes into text. image/* -> description, audio/* ->
    transcript, video/* -> description, application/pdf -> extracted text (or a
    vision read of rendered pages). Raises MediaUnsupported for other mimes,
    RuntimeError if no provider key is configured."""
    mime = (mime or "").lower().split(";")[0].strip()
    if mime == "application/pdf" or mime.endswith("/pdf"):
        return _understand_pdf(data, models=models, prompt=prompt, max_tokens=max(max_tokens, 1024))
    messages, default_models = _build(data, mime, prompt)
    return _run(messages, default_models, models, max_tokens)


__all__ = [
    "DEFAULT_AUDIO_MODELS", "DEFAULT_VIDEO_MODELS", "DEFAULT_VISION_MODELS",
    "MediaUnsupported", "understand_media",
]
