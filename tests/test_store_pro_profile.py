from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from supergraph import SuperGraph
from supergraph.pro import (
    CalibrationCache, CalibrationEntry, HostSnapshot, ProSpec,
    ProCalibrationMissing, ProExtraNotInstalled,
)


def _host_with_all_extras() -> HostSnapshot:
    return HostSnapshot(
        ram_total_mb=32768, ram_available_mb=24000, disk_free_mb=80000,
        cpu_cores_physical=8, cpu_cores_logical=16,
        gpu_ready=False, gpu_name=None,
        gpu_vram_total_mb=0, gpu_vram_free_mb=0,
        extras_installed=frozenset({
            "llama-cpp-python", "onnxruntime", "tokenizers",
            "huggingface-hub",
        }),
    )


def _entry(cid: str, **overrides) -> CalibrationEntry:
    base = CalibrationEntry(
        component_id=cid,
        measured_at=datetime.now(timezone.utc).isoformat(),
        ram_mb_idle=200, ram_mb_at_default=600,
        ram_mb_min=400, ram_mb_max=800,
        disk_mb=150, vram_mb_full_offload=0,
        tps_cpu_threads={"16": 80.0},
        tps_gpu_full_offload=None,
        extra={
            "n_ctx_min": 2048, "n_ctx_default": 4096, "n_ctx_max": 8192,
            "n_batch_at_default": 512, "embed_batch_at_default": 64,
            "reranker_max_at_default": 1024,
        },
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _seed_cache(cache_dir: Path, host: HostSnapshot, spec: ProSpec) -> None:
    cache = CalibrationCache.empty(host.host_signature())
    for cid in spec.component_ids():
        cache.components[cid] = _entry(cid)
    cache.save(cache_dir=cache_dir)


@pytest.fixture
def stub_host(monkeypatch):
    host = _host_with_all_extras()
    monkeypatch.setattr(
        HostSnapshot, "capture",
        classmethod(lambda cls, cache_dir=None, probe_gpu=True: host),
    )
    return host


class TestProfileKwarg:
    def test_default_profile_none_is_noop(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        try:
            assert gs.pro_spec is None
            assert gs.pro_resolved is None
        finally:
            gs.close()

    def test_unknown_profile_raises(self, tmp_path):
        with pytest.raises(ValueError, match="unknown profile"):
            SuperGraph(path=str(tmp_path / "db"), profile="ultra")

    def test_pro_spec_must_be_prospec_instance(self, tmp_path, stub_host):
        with pytest.raises(TypeError, match="ProSpec"):
            SuperGraph(
                path=str(tmp_path / "db"),
                profile="pro",
                pro_spec="default",
                pro_cache_dir=str(tmp_path / "cache"),
                pro_strict=False,
            )


class TestProCalibration:
    def test_strict_no_calibration_raises(self, tmp_path, stub_host):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        with pytest.raises(ProCalibrationMissing) as excinfo:
            SuperGraph(
                path=str(tmp_path / "db"),
                profile="pro",
                pro_cache_dir=str(cache_dir),
                pro_strict=True,
                embedder=None,
            )
        assert excinfo.value.missing_components, (
            "missing_components should list every component_id from the spec"
        )

    def test_lenient_no_calibration_warns_but_builds(self, tmp_path, stub_host, caplog):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        with caplog.at_level("WARNING"):
            gs = SuperGraph(
                path=str(tmp_path / "db"),
                profile="pro",
                pro_cache_dir=str(cache_dir),
                pro_strict=False,
                embedder=None,
            )
        try:
            assert gs.pro_resolved is not None
            assert gs.pro_resolved.fits is False
            assert "does not fit host" in caplog.text
        finally:
            gs.close()

    def test_fits_with_seeded_cache(self, tmp_path, stub_host):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        spec = ProSpec()
        _seed_cache(cache_dir, stub_host, spec)
        gs = SuperGraph(
            path=str(tmp_path / "db"),
            profile="pro",
            pro_spec=spec,
            pro_cache_dir=str(cache_dir),
            pro_strict=True,
            embedder=None,
        )
        try:
            assert gs.pro_spec is spec
            assert gs.pro_resolved is not None
            assert gs.pro_resolved.fits is True
            assert gs.pro_resolved.n_ctx == 8192
            assert gs.pro_resolved.bonsai_n_batch == 512
            assert gs.pro_resolved.bonsai_n_gpu_layers == 0
        finally:
            gs.close()


class TestProExtras:
    def test_strict_missing_extra_raises(self, tmp_path, monkeypatch):
        host = HostSnapshot(
            ram_total_mb=16000, ram_available_mb=12000, disk_free_mb=20000,
            cpu_cores_physical=8, cpu_cores_logical=16,
            gpu_ready=False, gpu_name=None,
            gpu_vram_total_mb=0, gpu_vram_free_mb=0,
            extras_installed=frozenset({"onnxruntime"}),
        )
        monkeypatch.setattr(
            HostSnapshot, "capture",
            classmethod(lambda cls, cache_dir=None, probe_gpu=True: host),
        )
        with pytest.raises(ProExtraNotInstalled):
            SuperGraph(
                path=str(tmp_path / "db"),
                profile="pro",
                pro_cache_dir=str(tmp_path / "cache"),
                pro_strict=True,
                embedder=None,
            )


class TestCreateBonsaiFactory:
    def test_requires_profile_pro(self, tmp_path):
        gs = SuperGraph(path=str(tmp_path / "db"), embedder=None)
        try:
            with pytest.raises(RuntimeError, match="profile='pro'"):
                gs.create_bonsai()
        finally:
            gs.close()

    def test_refuses_when_ingest_mode_not_bonsai(self, tmp_path, stub_host):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        spec = ProSpec(ingest_mode="deterministic")
        _seed_cache(cache_dir, stub_host, spec)
        gs = SuperGraph(
            path=str(tmp_path / "db"),
            profile="pro",
            pro_spec=spec,
            pro_cache_dir=str(cache_dir),
            embedder=None,
        )
        try:
            with pytest.raises(RuntimeError, match="ingest_mode"):
                gs.create_bonsai()
        finally:
            gs.close()

    def test_refuses_when_spec_does_not_fit(self, tmp_path, stub_host):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        gs = SuperGraph(
            path=str(tmp_path / "db"),
            profile="pro",
            pro_cache_dir=str(cache_dir),
            pro_strict=False,
            embedder=None,
        )
        try:
            with pytest.raises(RuntimeError, match="does not fit"):
                gs.create_bonsai()
        finally:
            gs.close()

    def test_locates_gguf_via_hf_cache(self, tmp_path, stub_host, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        spec = ProSpec()
        _seed_cache(cache_dir, stub_host, spec)

        fake_gguf = tmp_path / "Ternary-Bonsai-4B-TQ1_0.gguf"
        fake_gguf.write_bytes(b"\x00")

        class _FakeFile:
            def __init__(self, path: Path):
                self.file_name = path.name
                self.file_path = str(path)

        class _FakeRev:
            files = [_FakeFile(fake_gguf)]

        class _FakeRepo:
            repo_id = "superkaiii/Ternary-Bonsai-4B-GGUF"
            revisions = [_FakeRev()]

        class _FakeCache:
            repos = [_FakeRepo()]

        monkeypatch.setattr(
            "huggingface_hub.scan_cache_dir", lambda: _FakeCache(),
        )

        captured = {}
        from supergraph import bonsai_ingestor as _bi
        original = _bi.BonsaiIngestor

        class _StubIngestor:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(_bi, "BonsaiIngestor", _StubIngestor)
        gs = SuperGraph(
            path=str(tmp_path / "db"),
            profile="pro",
            pro_spec=spec,
            pro_cache_dir=str(cache_dir),
            embedder=None,
        )
        try:
            ing = gs.create_bonsai()
            assert isinstance(ing, _StubIngestor)
            assert captured["model_path"] == str(fake_gguf)
            assert captured["n_ctx"] == 8192
            assert captured["n_batch"] == 512
            assert captured["n_gpu_layers"] == 0
            assert captured["gs"] is gs
            assert captured["ner_model_dir"]
        finally:
            gs.close()
            monkeypatch.setattr(_bi, "BonsaiIngestor", original)
