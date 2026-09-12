from __future__ import annotations

import logging
from typing import Any

from supergraph.algos.chunker import Chunk
from supergraph.ingest.base import Ingestor, IngestResult

logger = logging.getLogger(__name__)

_model_cache: dict[tuple[str, str, str], Any] = {}

_DEFAULT_MODEL = "base"
_DEFAULT_DEVICE = "cpu"
_DEFAULT_COMPUTE_TYPE = "int8"


def _get_model(model_size: str, device: str, compute_type: str):
    key = (model_size, device, compute_type)
    if key not in _model_cache:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "WhisperIngestor requires the `stt` extra. "
                "Install with: pip install 'supergraph[audio]'"
            ) from e
        _model_cache[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _model_cache[key]


class WhisperIngestor(Ingestor):

    name = "whisper"
    supported_extensions = ["wav", "mp3", "ogg", "flac", "m4a", "opus", "webm"]

    def convert(self, file_path: str, **kwargs) -> IngestResult:
        model_size = kwargs.get("model", _DEFAULT_MODEL)
        device = kwargs.get("device", _DEFAULT_DEVICE)
        compute_type = kwargs.get("compute_type", _DEFAULT_COMPUTE_TYPE)
        language = kwargs.get("language")
        beam_size = kwargs.get("beam_size", 1)
        chunk_max_size = kwargs.get("max_chunk_size", 2000)
        summary_max_len = kwargs.get("summary_max_len", 200)

        model = _get_model(model_size, device, compute_type)
        segments_iter, info = model.transcribe(
            file_path,
            language=language,
            beam_size=beam_size,
            vad_filter=kwargs.get("vad_filter", True),
        )

        chunks: list[Chunk] = []
        parts: list[str] = []
        current_text: list[str] = []
        current_start: float | None = None
        current_end: float | None = None
        total_len = 0

        def _fmt_time(s: float | None) -> str:
            if s is None:
                return "?"
            m, sec = divmod(int(s), 60)
            return f"{m:02d}:{sec:02d}"

        def flush(idx: int):
            if not current_text:
                return
            text = " ".join(current_text).strip()
            if not text:
                return
            heading = f"[{_fmt_time(current_start)}-{_fmt_time(current_end)}]"
            chunks.append(Chunk(
                text=text,
                summary=text[:summary_max_len],
                index=idx,
                heading=heading,
                page=idx,
            ))

        for seg in segments_iter:
            seg_text = (seg.text or "").strip()
            if not seg_text:
                continue
            parts.append(seg_text)
            if current_start is None:
                current_start = seg.start
            current_end = seg.end
            current_text.append(seg_text)
            total_len += len(seg_text) + 1
            if total_len >= chunk_max_size:
                flush(len(chunks))
                current_text = []
                current_start = None
                current_end = None
                total_len = 0

        flush(len(chunks))

        transcript = "\n".join(parts).strip()

        duration_s = getattr(info, "duration", None)
        language_code = getattr(info, "language", None) or language
        language_prob = getattr(info, "language_probability", None)
        metadata = {
            "source": file_path,
            "model": model_size,
            "device": device,
            "compute_type": compute_type,
            "duration_s": duration_s,
            "language": language_code,
            "language_probability": language_prob,
        }

        confidence = 1.0
        if not transcript:
            confidence = 0.1
            metadata["warning"] = (
                "No speech detected. Audio may be silent or heavily noisy. "
                "Try model='small' or disable vad_filter."
            )
        elif language_prob is not None and language_prob < 0.5:
            confidence = max(0.3, language_prob)
            metadata["warning"] = (
                f"Low language-detection confidence ({language_prob:.2f}). "
                f"Consider passing language='<iso_code>' explicitly."
            )

        return IngestResult(
            markdown=transcript,
            chunks=chunks,
            images=[],
            metadata=metadata,
            parser_used=self.name,
            confidence=confidence,
        )
