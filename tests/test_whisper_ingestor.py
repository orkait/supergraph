from __future__ import annotations

import struct
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


FIXTURES = Path(__file__).parent / "fixtures" / "audio"


def _silent_wav(path: Path, seconds: float = 1.0, rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack(f"<{int(rate * seconds)}h", *([0] * int(rate * seconds))))
    return path


def _fake_model(segments, language="en", language_probability=0.95, duration=2.0):
    seg_objs = [SimpleNamespace(start=s, end=e, text=t) for s, e, t in segments]
    info = SimpleNamespace(
        language=language,
        language_probability=language_probability,
        duration=duration,
    )

    class FakeModel:
        def transcribe(self, *a, **k):
            return iter(seg_objs), info

    return FakeModel()


def test_whisper_ingestor_transcript_assembly(tmp_path):
    from supergraph.ingest import whisper_ingestor as wi
    wav = _silent_wav(tmp_path / "x.wav")
    fake = _fake_model([(0.0, 1.2, "hello"), (1.2, 2.0, "world")])
    with patch.object(wi, "_get_model", return_value=fake):
        ing = wi.WhisperIngestor()
        r = ing.convert(str(wav))
    assert r.parser_used == "whisper"
    assert "hello" in r.markdown
    assert "world" in r.markdown
    assert r.confidence == 1.0
    assert len(r.chunks) == 1
    assert r.chunks[0].heading == "[00:00-00:02]"
    assert r.metadata["language"] == "en"


def test_whisper_ingestor_empty_audio_sets_low_confidence(tmp_path):
    from supergraph.ingest import whisper_ingestor as wi
    wav = _silent_wav(tmp_path / "empty.wav")
    fake = _fake_model([], language_probability=0.2)
    with patch.object(wi, "_get_model", return_value=fake):
        ing = wi.WhisperIngestor()
        r = ing.convert(str(wav))
    assert r.markdown == ""
    assert r.confidence < 0.5
    assert "warning" in r.metadata


def test_whisper_ingestor_low_language_prob_warning(tmp_path):
    from supergraph.ingest import whisper_ingestor as wi
    wav = _silent_wav(tmp_path / "lang.wav")
    fake = _fake_model([(0.0, 1.0, "garbled text")], language_probability=0.3)
    with patch.object(wi, "_get_model", return_value=fake):
        ing = wi.WhisperIngestor()
        r = ing.convert(str(wav))
    assert r.confidence < 0.5
    assert "language" in r.metadata["warning"].lower()


def test_router_routes_audio_exts_when_audio_installed():
    pytest.importorskip("faster_whisper")
    from supergraph.ingest.router import EXTENSION_MAP
    assert EXTENSION_MAP.get("wav") == "whisper"
    assert EXTENSION_MAP.get("mp3") == "whisper"
    assert EXTENSION_MAP.get("m4a") == "whisper"
    assert EXTENSION_MAP.get("flac") == "whisper"


def test_router_skips_audio_exts_when_audio_missing():
    try:
        import faster_whisper  # noqa: F401
        pytest.skip("faster_whisper installed; this test checks absence behaviour")
    except ImportError:
        from supergraph.ingest.router import EXTENSION_MAP
        assert EXTENSION_MAP.get("wav") is None
        assert EXTENSION_MAP.get("mp3") is None


def test_list_ingestors_includes_whisper_with_audio_extra():
    from supergraph.ingest.router import list_ingestors
    entry = next((i for i in list_ingestors() if i["name"] == "whisper"), None)
    assert entry is not None
    assert entry["extra"] == "audio"
    assert "wav" in entry["formats"]
    assert "flac" in entry["formats"]


def test_whisper_ingestor_surfaces_missing_extra(tmp_path, monkeypatch):
    from supergraph.ingest import whisper_ingestor as wi
    wav = _silent_wav(tmp_path / "x.wav")
    wi._model_cache.clear()
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "faster_whisper":
            raise ImportError("no")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ing = wi.WhisperIngestor()
    with pytest.raises(ImportError, match=r"supergraphdb\[audio\]"):
        ing.convert(str(wav))


@pytest.mark.needs_audio
@pytest.mark.parametrize("fixture_name,expected_keywords", [
    ("jfk_inaugural_11s.wav", ["americans", "ask", "country"]),
    ("librispeech_stew_dinner_10s.flac", ["stew", "dinner"]),
    ("librispeech_yellow_lamps_7s.flac", ["yellow", "lamp"]),
])
def test_real_whisper_transcription(fixture_name, expected_keywords):
    clip = FIXTURES / fixture_name
    if not clip.exists():
        pytest.skip(f"fixture missing: {clip}")
    try:
        import faster_whisper  # noqa
    except ImportError:
        pytest.skip("supergraphdb[audio] not installed")
    from supergraph.ingest.whisper_ingestor import WhisperIngestor
    ing = WhisperIngestor()
    r = ing.convert(str(clip), model="tiny")
    assert r.parser_used == "whisper"
    assert r.confidence >= 0.5, f"low confidence: {r.confidence} txt={r.markdown!r}"
    lowered = r.markdown.lower()
    hits = [kw for kw in expected_keywords if kw in lowered]
    assert hits, f"no expected keywords in transcript: {r.markdown!r}"


@pytest.mark.needs_audio
def test_real_router_dispatches_wav_to_whisper():
    clip = FIXTURES / "jfk_inaugural_11s.wav"
    if not clip.exists():
        pytest.skip(f"fixture missing: {clip}")
    try:
        import faster_whisper  # noqa
    except ImportError:
        pytest.skip("supergraphdb[audio] not installed")
    from supergraph.ingest.router import ingest_file
    r = ingest_file(str(clip))
    assert r.parser_used == "whisper"
    assert "country" in r.markdown.lower() or "american" in r.markdown.lower()
    assert len(r.chunks) >= 1
