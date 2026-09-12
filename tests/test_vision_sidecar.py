from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def vlm_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPERGRAPH_VLM_CACHE_DIR", str(tmp_path))
    yield tmp_path


def test_status_when_no_pid_file(vlm_cache):
    from supergraph.ingest import vision_sidecar as vs
    st = vs.status()
    assert st.running is False
    assert st.pid is None
    assert st.base_url is None


def test_read_pid_ignores_dead_process(vlm_cache):
    from supergraph.ingest import vision_sidecar as vs
    (vlm_cache / "sidecar.pid").write_text(json.dumps({"pid": 999999, "port": 8418, "model": "x"}))
    assert vs._read_pid() is None


def test_read_pid_preserves_live_process(vlm_cache):
    from supergraph.ingest import vision_sidecar as vs
    (vlm_cache / "sidecar.pid").write_text(json.dumps({"pid": os.getpid(), "port": 8418, "model": "foo.gguf"}))
    pid, port, model = vs._read_pid()
    assert pid == os.getpid()
    assert port == 8418
    assert model == "foo.gguf"


def test_read_pid_ignores_corrupt_file(vlm_cache):
    from supergraph.ingest import vision_sidecar as vs
    (vlm_cache / "sidecar.pid").write_text("{not json")
    assert vs._read_pid() is None


def test_resolve_base_url_prefers_env(vlm_cache, monkeypatch):
    from supergraph.ingest import vision_sidecar as vs
    monkeypatch.setenv("SUPERGRAPH_VISION_URL", "http://custom.invalid:9/v1")
    assert vs.resolve_base_url() == "http://custom.invalid:9/v1"


def test_resolve_base_url_no_autostart_returns_none(vlm_cache, monkeypatch):
    from supergraph.ingest import vision_sidecar as vs
    monkeypatch.delenv("SUPERGRAPH_VISION_URL", raising=False)
    with patch.object(vs, "_probe", return_value=False):
        assert vs.resolve_base_url(auto_start=False) is None


def test_resolve_base_url_uses_external_sidecar_if_probing(vlm_cache, monkeypatch):
    from supergraph.ingest import vision_sidecar as vs
    monkeypatch.delenv("SUPERGRAPH_VISION_URL", raising=False)
    with patch.object(vs, "_probe", return_value=True):
        url = vs.resolve_base_url(auto_start=False)
    assert url == "http://127.0.0.1:8418/v1"


def test_start_reuses_live_sidecar(vlm_cache):
    from supergraph.ingest import vision_sidecar as vs
    pid = os.getpid()
    (vlm_cache / "sidecar.pid").write_text(json.dumps({"pid": pid, "port": 8418, "model": "a.gguf"}))
    with patch.object(vs, "_probe", return_value=True):
        st = vs.start(wait_ready=False)
    assert st.running is True
    assert st.pid == pid
    assert st.port == 8418


def test_start_spawns_subprocess(vlm_cache):
    from supergraph.ingest import vision_sidecar as vs
    fake_proc = MagicMock()
    fake_proc.pid = 424242
    fake_proc.poll.return_value = None
    with patch.object(vs, "download_weights", return_value=(vlm_cache / "m.gguf", vlm_cache / "mm.gguf")):
        with patch.object(vs, "_probe", side_effect=[False, True]):
            with patch("subprocess.Popen", return_value=fake_proc) as popen:
                st = vs.start(wait_ready=True, ready_timeout=5.0)
    assert popen.called
    assert st.pid == 424242
    rec = json.loads((vlm_cache / "sidecar.pid").read_text())
    assert rec["pid"] == 424242


def test_stop_removes_pid_file(vlm_cache):
    from supergraph.ingest import vision_sidecar as vs
    (vlm_cache / "sidecar.pid").write_text(json.dumps({"pid": os.getpid(), "port": 8418, "model": "x"}))
    calls = {"n": 0}

    def fake_kill(pid, sig):
        calls["n"] += 1
        if calls["n"] <= 2:
            return None
        raise ProcessLookupError()

    with patch("os.kill", side_effect=fake_kill):
        assert vs.stop() is True
    assert not (vlm_cache / "sidecar.pid").exists()


def test_stop_returns_false_when_not_running(vlm_cache):
    from supergraph.ingest import vision_sidecar as vs
    assert vs.stop() is False


def test_resolve_spec_defaults_to_smolvlm2_2_2b(vlm_cache, monkeypatch):
    from supergraph.ingest import vision_sidecar as vs
    monkeypatch.delenv("SUPERGRAPH_VISION_MODEL", raising=False)
    spec = vs.resolve_spec(None)
    assert spec.repo == "ggml-org/SmolVLM2-2.2B-Instruct-GGUF"


def test_resolve_spec_reads_env(vlm_cache, monkeypatch):
    from supergraph.ingest import vision_sidecar as vs
    monkeypatch.setenv("SUPERGRAPH_VISION_MODEL", "smolvlm-500m")
    spec = vs.resolve_spec(None)
    assert spec.repo == "ggml-org/SmolVLM-500M-Instruct-GGUF"


def test_resolve_spec_passthrough_object(vlm_cache):
    from supergraph.ingest import vision_sidecar as vs
    custom = vs.VLMModelSpec(repo="x/y", model_file="m.gguf", mmproj_file="mm.gguf")
    assert vs.resolve_spec(custom) is custom


def test_resolve_spec_unknown_preset_raises(vlm_cache):
    from supergraph.ingest import vision_sidecar as vs
    with pytest.raises(KeyError, match="Unknown VLM preset"):
        vs.resolve_spec("no-such-preset")


def test_register_model_adds_preset(vlm_cache):
    from supergraph.ingest import vision_sidecar as vs
    spec = vs.VLMModelSpec(repo="a/b", model_file="c.gguf", mmproj_file="d.gguf")
    vs.register_model("tmp-test", spec)
    try:
        assert vs.resolve_spec("tmp-test") is spec
    finally:
        vs.VLM_MODELS.pop("tmp-test", None)


def test_env_port_override(vlm_cache, monkeypatch):
    from supergraph.ingest import vision_sidecar as vs
    monkeypatch.setenv("SUPERGRAPH_VISION_PORT", "9999")
    assert vs._env_port() == 9999


def test_env_port_invalid_falls_back(vlm_cache, monkeypatch):
    from supergraph.ingest import vision_sidecar as vs
    monkeypatch.setenv("SUPERGRAPH_VISION_PORT", "not-a-number")
    assert vs._env_port() == vs.DEFAULT_PORT


def test_env_host_override(vlm_cache, monkeypatch):
    from supergraph.ingest import vision_sidecar as vs
    monkeypatch.setenv("SUPERGRAPH_VISION_HOST", "0.0.0.0")
    assert vs._env_host() == "0.0.0.0"


def test_download_weights_surfaces_missing_extra(vlm_cache, monkeypatch):
    from supergraph.ingest import vision_sidecar as vs
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "huggingface_hub":
            raise ImportError("no hf")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="supergraphdb\\[vision\\]"):
        vs.download_weights()
