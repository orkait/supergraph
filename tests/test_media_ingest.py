"""supergraph owns multimodal understanding: media bytes -> text -> stored.
Covers all required modalities: image / audio / video / pdf."""
import base64

import pytest

import supergraph.ingest.media as media


def test_image_builds_image_part_with_vision_models():
    msgs, models = media._build(b"\xff\xd8\xff", "image/jpeg", None)
    parts = msgs[0]["content"]
    assert any(p["type"] == "image_url" and p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in parts)
    assert models == media.DEFAULT_VISION_MODELS


def test_audio_builds_input_audio_part_with_mapped_format():
    msgs, models = media._build(b"ID3", "audio/mpeg", None)  # mpeg -> mp3
    parts = msgs[0]["content"]
    assert any(p["type"] == "input_audio" and p["input_audio"]["format"] == "mp3" for p in parts)
    assert models == media.DEFAULT_AUDIO_MODELS


def test_video_builds_video_url_part():
    msgs, models = media._build(b"\x00\x00\x00\x18ftyp", "video/mp4", None)
    parts = msgs[0]["content"]
    assert any(p["type"] == "video_url" and p["video_url"]["url"].startswith("data:video/mp4;base64,") for p in parts)
    assert models == media.DEFAULT_VIDEO_MODELS


def test_unsupported_mime_raises():
    with pytest.raises(media.MediaUnsupported):
        media._build(b"PK", "application/zip", None)


def test_pdf_with_text_returns_extracted_text_no_llm(monkeypatch):
    # text PDF: understanding is local extraction, no model call
    monkeypatch.setattr(media, "_pdf_text", lambda data: "Quarterly revenue grew 12 percent in Q3.")
    monkeypatch.setattr(media, "_run", lambda *a, **k: pytest.fail("must not call the LLM for a text PDF"))
    out = media.understand_media(b"%PDF-fake", "application/pdf")
    assert out == "Quarterly revenue grew 12 percent in Q3."


def test_scanned_pdf_falls_back_to_vision(monkeypatch):
    monkeypatch.setattr(media, "_pdf_text", lambda data: "")          # no extractable text
    monkeypatch.setattr(media, "_pdf_page_pngs", lambda data, n: [b"\x89PNG-page1"])
    monkeypatch.setattr(media, "_run", lambda msgs, dm, m, mt: "a scanned invoice for $400")
    out = media.understand_media(b"%PDF-scan", "application/pdf")
    assert out == "a scanned invoice for $400"


def test_ingest_media_endpoint_understands_and_stores(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import supergraph.server as server
    monkeypatch.setattr(media, "understand_media",
                        lambda data, mime, **kw: "a red bicycle leaning on a brick wall")
    server._store = None
    with TestClient(server.app) as client:
        img_b64 = base64.b64encode(b"\xff\xd8\xff\xe0fake").decode()
        r = client.post("/api/ingest-media", json={
            "id": "m:1", "mime": "image/jpeg", "data_b64": img_b64, "namespace": "ns",
        }).json()
        assert r["kind"] == "media" and r["data"]["text"] == "a red bicycle leaning on a brick wall"
        rec = client.post("/api/execute", json={"query": 'REMEMBER "bicycle" LIMIT 3', "namespace": "ns"}).json()
        assert "m:1" in [row.get("id") for row in (rec.get("data") or [])]


def test_ingest_media_endpoint_rejects_bad_base64(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import supergraph.server as server
    server._store = None
    with TestClient(server.app) as client:
        r = client.post("/api/ingest-media", json={"id": "m:2", "mime": "image/jpeg", "data_b64": "!!!notb64"})
        assert r.json()["kind"] == "error"
