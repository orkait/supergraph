"""Tests for supergraph.pro: ProSpec, HostSnapshot, calibration cache,
resolve(). Live calibration probing is exercised by PR#3's CLI tests;
this file covers the deterministic parts.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
import msgspec

from supergraph.pro import (
    CalibrationCache, CalibrationEntry,
    HostSnapshot, ProSpec, ResolvedConfig,
    ProCalibrationMissing, ProExtraNotInstalled, ProUnsupportedHostError,
    check_extras_installed, resolve,
)


# ---------------------------------------------------------------------
# ProSpec
# ---------------------------------------------------------------------


class TestProSpec:
    def test_defaults_match_design(self):
        s = ProSpec()
        assert s.embedder == "jina-v5-small"
        assert s.reranker == "jina-v3"
        assert s.ingest_mode == "bonsai"
        assert s.bonsai_quant == "tq1_0"
        assert s.bonsai_skill == "lite"
        assert s.vision == "none"
        assert s.audio == "none"
        assert s.ner == "tinybert"

    def test_frozen(self):
        s = ProSpec()
        with pytest.raises(AttributeError):
            s.embedder = "model2vec-256d"

    def test_component_ids_default_set(self):
        ids = ProSpec().component_ids()
        assert "embedder:jina-v5-small" in ids
        assert "reranker:jina-v3" in ids
        assert "ingest:bonsai-tq1_0-lite" in ids
        assert "ner:tinybert" in ids
        assert all("vision:" not in i for i in ids)
        assert all("audio:" not in i for i in ids)

    def test_component_ids_deterministic_skips_bonsai(self):
        ids = ProSpec(ingest_mode="deterministic").component_ids()
        assert all("ingest:" not in i for i in ids)

    def test_component_ids_with_vision_audio(self):
        ids = ProSpec(vision="smolvlm2-2.2b", audio="whisper-base").component_ids()
        assert "vision:smolvlm2-2.2b" in ids
        assert "audio:whisper-base" in ids

    def test_required_dists_includes_llama_cpp_for_bonsai(self):
        assert "llama-cpp-python" in ProSpec().required_dists()

    def test_required_dists_drops_llama_cpp_for_deterministic_no_vision(self):
        s = ProSpec(ingest_mode="deterministic", vision="none")
        assert "llama-cpp-python" not in s.required_dists()

    def test_required_dists_dedups(self):
        s = ProSpec()  # ner=tinybert and embedder=jina-v5-small both want onnxruntime
        assert s.required_dists().count("onnxruntime") == 1

    def test_msgspec_roundtrip(self):
        s = ProSpec(reranker="none", vision="qwen-vl-3b")
        encoded = msgspec.json.encode(s)
        decoded = msgspec.json.decode(encoded, type=ProSpec)
        assert decoded == s


# ---------------------------------------------------------------------
# HostSnapshot
# ---------------------------------------------------------------------


class TestHostSnapshot:
    def test_capture_returns_real_numbers(self, tmp_path):
        # No mocking - capture against the actual host. Numbers must be
        # plausible (positive RAM/disk; cores >= 1).
        snap = HostSnapshot.capture(cache_dir=tmp_path, probe_gpu=False)
        assert snap.ram_total_mb > 0
        assert snap.ram_available_mb > 0
        assert snap.disk_free_mb > 0
        assert snap.cpu_cores_logical >= 1
        assert snap.cpu_cores_physical >= 1
        # gpu_ready=False because probe_gpu=False; vram fields zero.
        assert snap.gpu_ready is False
        assert snap.gpu_vram_total_mb == 0

    def test_host_signature_stable_across_capture(self, tmp_path):
        s1 = HostSnapshot.capture(cache_dir=tmp_path, probe_gpu=False)
        s2 = HostSnapshot.capture(cache_dir=tmp_path, probe_gpu=False)
        # Same machine, same probe → same signature.
        assert s1.host_signature() == s2.host_signature()

    def test_host_signature_includes_no_gpu_marker(self):
        snap = HostSnapshot(
            ram_total_mb=8192, ram_available_mb=4000, disk_free_mb=10000,
            cpu_cores_physical=4, cpu_cores_logical=8,
            gpu_ready=False, gpu_name=None,
            gpu_vram_total_mb=0, gpu_vram_free_mb=0,
            extras_installed=frozenset(),
        )
        assert "gpu_none" in snap.host_signature()
        assert "cpu_4c8t" in snap.host_signature()

    def test_host_signature_changes_with_gpu_change(self):
        a = HostSnapshot(
            ram_total_mb=16384, ram_available_mb=10000, disk_free_mb=20000,
            cpu_cores_physical=8, cpu_cores_logical=16,
            gpu_ready=True, gpu_name="RTX 3060", gpu_vram_total_mb=12288,
            gpu_vram_free_mb=11000, extras_installed=frozenset(),
        )
        b = HostSnapshot(
            ram_total_mb=16384, ram_available_mb=10000, disk_free_mb=20000,
            cpu_cores_physical=8, cpu_cores_logical=16,
            gpu_ready=True, gpu_name="RTX 4090", gpu_vram_total_mb=24576,
            gpu_vram_free_mb=23000, extras_installed=frozenset(),
        )
        assert a.host_signature() != b.host_signature()


# ---------------------------------------------------------------------
# CalibrationCache
# ---------------------------------------------------------------------


def _entry(cid: str, **overrides) -> CalibrationEntry:
    """Build a calibration entry with realistic `extra` knob fields.

    The resolver reads knob values (n_ctx, n_batch, embed_batch,
    reranker_max) from `extra` because those are recorded by the probe
    at measurement time. Tests populate them so resolver picks the
    measured-default tier rather than the always-safe fallback.
    """
    extra = {
        "n_ctx_min": 2048,
        "n_ctx_default": 4096,
        "n_ctx_max": 8192,
        "n_batch_at_default": 512,
        "embed_batch_at_default": 128,
        "reranker_max_at_default": 2048,
    }
    base = CalibrationEntry(
        component_id=cid,
        measured_at=datetime.now(timezone.utc).isoformat(),
        ram_mb_idle=200,
        ram_mb_at_default=400,
        ram_mb_min=400,
        ram_mb_max=500,
        disk_mb=150,
        vram_mb_full_offload=100,
        tps_cpu_threads={"4": 50.0, "8": 90.0},
        tps_gpu_full_offload=300.0,
        extra=extra,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


class TestCalibrationCache:
    def test_load_missing_returns_empty(self, tmp_path):
        cache = CalibrationCache.load("test-sig", cache_dir=tmp_path)
        assert cache.components == {}
        assert cache.host_signature == "test-sig"

    def test_load_invalid_json_returns_empty(self, tmp_path):
        (tmp_path / "calibration.json").write_text("{not json")
        cache = CalibrationCache.load("test-sig", cache_dir=tmp_path)
        assert cache.components == {}

    def test_load_schema_mismatch_returns_empty(self, tmp_path):
        (tmp_path / "calibration.json").write_text(json.dumps({
            "schema_version": 9999,
            "supergraph_version": "0.5.0",
            "host_signature": "test-sig",
            "measured_at": "2026-05-02T00:00:00+00:00",
            "components": {"x": {}},
        }))
        cache = CalibrationCache.load("test-sig", cache_dir=tmp_path)
        assert cache.components == {}

    def test_load_host_signature_mismatch_discards(self, tmp_path):
        from supergraph import __version__ as gs_v
        (tmp_path / "calibration.json").write_text(json.dumps({
            "schema_version": 1,
            "supergraph_version": gs_v,
            "host_signature": "other-host",
            "measured_at": "2026-05-02T00:00:00+00:00",
            "components": {"embedder:jina-v5-small": {
                "measured_at": "x", "ram_mb_idle": 100, "ram_mb_at_default": 200,
                "ram_mb_min": 100, "ram_mb_max": 300, "disk_mb": 150,
                "vram_mb_full_offload": 0, "tps_cpu_threads": {},
                "tps_gpu_full_offload": None, "extra": {},
            }},
        }))
        cache = CalibrationCache.load("test-sig", cache_dir=tmp_path)
        assert cache.components == {}

    def test_save_then_load_roundtrip(self, tmp_path):
        cache = CalibrationCache.empty("test-sig")
        cache.components["embedder:jina-v5-small"] = _entry("embedder:jina-v5-small")
        path = cache.save(cache_dir=tmp_path)
        assert path.exists()

        loaded = CalibrationCache.load("test-sig", cache_dir=tmp_path)
        assert "embedder:jina-v5-small" in loaded.components
        e = loaded.components["embedder:jina-v5-small"]
        assert e.ram_mb_idle == 200
        assert e.tps_cpu_threads["8"] == 90.0
        assert e.tps_gpu_full_offload == 300.0


# ---------------------------------------------------------------------
# check_extras_installed
# ---------------------------------------------------------------------


class TestCheckExtras:
    def _host(self, installed: set[str]):
        return HostSnapshot(
            ram_total_mb=16384, ram_available_mb=12000, disk_free_mb=20000,
            cpu_cores_physical=8, cpu_cores_logical=16,
            gpu_ready=False, gpu_name=None,
            gpu_vram_total_mb=0, gpu_vram_free_mb=0,
            extras_installed=frozenset(installed),
        )

    def test_passes_when_all_installed(self):
        spec = ProSpec()
        host = self._host({"llama-cpp-python", "onnxruntime", "tokenizers"})
        check_extras_installed(spec, host)  # must not raise

    def test_raises_when_missing(self):
        spec = ProSpec()
        host = self._host({"onnxruntime"})  # missing llama-cpp-python, tokenizers
        with pytest.raises(ProExtraNotInstalled) as excinfo:
            check_extras_installed(spec, host)
        assert "llama-cpp-python" in excinfo.value.missing_dists
        assert "supergraph[pro]" in str(excinfo.value)


# ---------------------------------------------------------------------
# resolve()
# ---------------------------------------------------------------------


class TestResolve:
    def _gpu_host(self, ram_avail=12000, vram_free=11000):
        return HostSnapshot(
            ram_total_mb=16384, ram_available_mb=ram_avail, disk_free_mb=50000,
            cpu_cores_physical=8, cpu_cores_logical=16,
            gpu_ready=True, gpu_name="RTX 3060",
            gpu_vram_total_mb=12288, gpu_vram_free_mb=vram_free,
            extras_installed=frozenset({"llama-cpp-python", "onnxruntime", "tokenizers"}),
        )

    def _cpu_host(self, ram_avail=12000):
        return HostSnapshot(
            ram_total_mb=16384, ram_available_mb=ram_avail, disk_free_mb=50000,
            cpu_cores_physical=8, cpu_cores_logical=16,
            gpu_ready=False, gpu_name=None,
            gpu_vram_total_mb=0, gpu_vram_free_mb=0,
            extras_installed=frozenset({"llama-cpp-python", "onnxruntime", "tokenizers"}),
        )

    def _full_cache(self, host):
        cache = CalibrationCache.empty(host.host_signature())
        for cid in ProSpec().component_ids():
            cache.components[cid] = _entry(cid,
                ram_mb_idle=200, ram_mb_at_default=600,
                ram_mb_min=400, ram_mb_max=800,
                vram_mb_full_offload=400 if "ingest:" in cid else 200,
            )
        return cache

    def test_calibration_missing_returns_fits_false(self, tmp_path):
        host = self._cpu_host()
        empty = CalibrationCache.empty(host.host_signature())
        rc = resolve(ProSpec(), host=host, cache=empty, cache_dir=tmp_path)
        assert rc.fits is False
        assert rc.calibration_source == "missing"
        assert any("calibration missing" in s for s in rc.shortfalls)

    def test_fits_with_full_cache_cpu_host(self, tmp_path):
        host = self._cpu_host()
        cache = self._full_cache(host)
        rc = resolve(ProSpec(), host=host, cache=cache, cache_dir=tmp_path)
        assert rc.fits is True
        assert rc.calibration_source == "measured"
        assert rc.bonsai_n_gpu_layers == 0  # CPU host
        # GPU-not-detected warning when bonsai is selected.
        assert any("GPU not detected" in w for w in rc.warnings)

    def test_fits_with_gpu_offloads_bonsai(self, tmp_path):
        host = self._gpu_host()
        cache = self._full_cache(host)
        rc = resolve(ProSpec(), host=host, cache=cache, cache_dir=tmp_path)
        assert rc.fits is True
        assert rc.bonsai_n_gpu_layers == -1
        # Larger embed_batch on GPU.
        assert rc.embed_batch == 128

    def test_fits_false_when_disk_short(self, tmp_path):
        host = self._cpu_host()
        host = HostSnapshot(**{**host.__dict__, "disk_free_mb": 10})  # tight disk
        cache = self._full_cache(host)
        rc = resolve(ProSpec(), host=host, cache=cache, cache_dir=tmp_path)
        assert rc.fits is False
        assert any("disk" in s.lower() for s in rc.shortfalls)

    def test_fits_false_when_ram_short(self, tmp_path):
        host = self._cpu_host(ram_avail=200)  # below sum of components
        cache = self._full_cache(host)
        rc = resolve(ProSpec(), host=host, cache=cache, cache_dir=tmp_path)
        assert rc.fits is False
        assert any("RAM" in s for s in rc.shortfalls)
        assert rc.suggestions  # at least one drop suggestion

    def test_layered_vram_falls_back_to_cpu(self, tmp_path):
        # Tiny VRAM forces layered allocation; only bonsai should keep
        # GPU; reranker drops to CPU.
        host = self._gpu_host(vram_free=500)  # enough for bonsai (400) only
        cache = self._full_cache(host)
        rc = resolve(ProSpec(), host=host, cache=cache, cache_dir=tmp_path)
        assert rc.fits is True
        assert any("layered" in w.lower() for w in rc.warnings)
        assert rc.bonsai_n_gpu_layers == -1
        assert rc.reranker_gpu_layers == 0

    def test_aged_calibration_emits_warning(self, tmp_path):
        host = self._gpu_host()
        cache = self._full_cache(host)
        # Backdate measured_at on every entry to 60 days ago.
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        cache.measured_at = old
        rc = resolve(ProSpec(), host=host, cache=cache, cache_dir=tmp_path)
        assert any("days old" in w for w in rc.warnings)

    def test_knobs_read_from_cache_extra_not_hardcoded(self, tmp_path):
        """Resolver must take n_ctx / n_batch / embed_batch / reranker_max
        from the calibration entry's `extra` dict, never from hard-coded
        host-RAM thresholds. The hard rule for pro is no assumption-based
        sizing; everything either measured or fallback."""
        host = self._gpu_host()
        cache = self._full_cache(host)
        # Override the bonsai entry's extra to non-default values; the
        # resolver should pick them up verbatim (within the fits ladder).
        bonsai_id = next(c for c in ProSpec().component_ids()
                         if c.startswith("ingest:bonsai-"))
        e = cache.components[bonsai_id]
        e.extra = dict(e.extra)
        e.extra["n_ctx_max"] = 16384
        e.extra["n_batch_at_default"] = 768
        cache.components[bonsai_id] = e
        # Embedder override
        emb_id = "embedder:jina-v5-small"
        cache.components[emb_id].extra = {"embed_batch_at_default": 64}
        # Reranker override
        rr_id = "reranker:jina-v3"
        cache.components[rr_id].extra = {"reranker_max_at_default": 1500}

        rc = resolve(ProSpec(), host=host, cache=cache, cache_dir=tmp_path)
        assert rc.fits is True
        assert rc.n_ctx == 16384
        assert rc.bonsai_n_batch == 768
        assert rc.embed_batch == 64
        assert rc.reranker_max_len == 1500

    def test_empty_extra_falls_back_safely(self, tmp_path):
        """When cache has no extra knobs (e.g. legacy probe), resolver
        uses safe minimums - never explodes, never picks unmeasured
        large values."""
        host = self._gpu_host()
        cache = self._full_cache(host)
        for cid in cache.components:
            cache.components[cid].extra = {}
        rc = resolve(ProSpec(), host=host, cache=cache, cache_dir=tmp_path)
        assert rc.fits is True
        assert rc.n_ctx == 2048      # _FALLBACK_N_CTX
        assert rc.bonsai_n_batch == 256  # _FALLBACK_N_BATCH
        assert rc.embed_batch == 16
        assert rc.reranker_max_len == 512
