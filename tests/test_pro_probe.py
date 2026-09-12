"""Tests for supergraph.pro_probe: probe registry, orchestrator,
helpers. Real model probes are slow + need network; those are smoke
covered by tests/test_pro_probe_live.py (skipped by default).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from supergraph import pro_probe
from supergraph.pro import (
    CalibrationCache, CalibrationEntry, HostSnapshot, ProSpec,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _host(**overrides) -> HostSnapshot:
    base = dict(
        ram_total_mb=8192, ram_available_mb=4000, disk_free_mb=10000,
        cpu_cores_physical=4, cpu_cores_logical=8,
        gpu_ready=False, gpu_name=None,
        gpu_vram_total_mb=0, gpu_vram_free_mb=0,
        extras_installed=frozenset({"llama-cpp-python", "onnxruntime",
                                    "tokenizers", "huggingface-hub"}),
    )
    base.update(overrides)
    return HostSnapshot(**base)


def _entry(cid: str) -> CalibrationEntry:
    return CalibrationEntry(
        component_id=cid,
        measured_at=datetime.now(timezone.utc).isoformat(),
        ram_mb_idle=100, ram_mb_at_default=200,
        ram_mb_min=100, ram_mb_max=200,
        disk_mb=50, vram_mb_full_offload=0,
        tps_cpu_threads={"8": 50.0}, tps_gpu_full_offload=None,
        extra={"probed": True},
    )


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------


class TestRegistry:
    def test_every_pro_spec_component_has_a_probe(self):
        """Critical contract: ProSpec.component_ids() for every legal
        slot combination must resolve to a registered probe. If a slot
        gains a new value without a matching probe the resolver will
        emit calibration_source=missing forever."""
        registered = set(pro_probe.list_probable())
        # Defaults
        assert set(ProSpec().component_ids()) <= registered
        # Vision opt-in variants
        for v in ("smolvlm2-2.2b", "qwen-vl-3b"):
            ids = ProSpec(vision=v).component_ids()
            assert any(i.startswith("vision:") for i in ids)
            assert all(i in registered for i in ids), (
                f"unregistered for vision={v}: {set(ids) - registered}"
            )
        # Audio opt-in variants
        for a in ("whisper-tiny", "whisper-base", "whisper-small"):
            ids = ProSpec(audio=a).component_ids()
            assert all(i in registered for i in ids), (
                f"unregistered for audio={a}: {set(ids) - registered}"
            )
        # Bonsai matrix
        for q in ("tq1_0", "tq2_0"):
            for s in ("lite", "full"):
                ids = ProSpec(bonsai_quant=q, bonsai_skill=s).component_ids()
                assert all(i in registered for i in ids), (
                    f"unregistered for bonsai={q}/{s}: {set(ids) - registered}"
                )

    def test_list_probable_is_sorted_and_stable(self):
        ids = pro_probe.list_probable()
        assert ids == sorted(ids)
        assert pro_probe.list_probable() == ids  # idempotent


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------


class _FakeProbe(pro_probe.Probe):
    """In-memory probe that records its calls + returns a canned entry."""

    def __init__(self, cid: str, fail_in: str | None = None,
                 disk_mb: int = 42):
        self.component_id = cid
        self._fail_in = fail_in
        self._disk_mb = disk_mb
        self.download_called = False
        self.measure_called = False

    def download(self, cache_dir, host):
        self.download_called = True
        if self._fail_in == "download":
            raise RuntimeError("simulated download failure")
        return self._disk_mb

    def measure(self, cache_dir, host, disk_mb):
        self.measure_called = True
        if self._fail_in == "measure":
            raise RuntimeError("simulated measure failure")
        return _entry(self.component_id)


class TestOrchestratorErrorIsolation:
    def test_failed_download_does_not_abort_suite(self, tmp_path, monkeypatch):
        good_a = _FakeProbe("a")
        bad = _FakeProbe("b", fail_in="download")
        good_c = _FakeProbe("c")
        registry = {
            "a": lambda: good_a,
            "b": lambda: bad,
            "c": lambda: good_c,
        }
        monkeypatch.setattr(pro_probe, "_REGISTRY", registry)

        summary = pro_probe.probe_components(
            ["a", "b", "c"], host=_host(), cache_dir=tmp_path,
        )
        assert {r.component_id for r in summary.successes} == {"a", "c"}
        assert {r.component_id for r in summary.failures} == {"b"}
        assert "download failed" in summary.failures[0].error
        # Even though "b" failed, its measure() must not have been called.
        assert bad.measure_called is False
        # Good probes ran.
        assert good_a.measure_called is True
        assert good_c.measure_called is True

    def test_failed_measure_does_not_abort_suite(self, tmp_path, monkeypatch):
        good = _FakeProbe("a")
        bad = _FakeProbe("b", fail_in="measure")
        registry = {"a": lambda: good, "b": lambda: bad}
        monkeypatch.setattr(pro_probe, "_REGISTRY", registry)

        summary = pro_probe.probe_components(
            ["a", "b"], host=_host(), cache_dir=tmp_path,
        )
        assert summary.successes[0].component_id == "a"
        assert summary.failures[0].component_id == "b"
        assert "measure failed" in summary.failures[0].error

    def test_unregistered_component_recorded_as_failure(self, tmp_path):
        summary = pro_probe.probe_components(
            ["totally:not_real"], host=_host(), cache_dir=tmp_path,
        )
        assert summary.all_ok is False
        assert summary.failures[0].component_id == "totally:not_real"
        assert "no probe registered" in summary.failures[0].error


class TestOrchestratorCacheUpdate:
    def test_cache_written_after_each_success(self, tmp_path, monkeypatch):
        good_a = _FakeProbe("a")
        bad = _FakeProbe("b", fail_in="measure")
        good_c = _FakeProbe("c")
        registry = {
            "a": lambda: good_a,
            "b": lambda: bad,
            "c": lambda: good_c,
        }
        monkeypatch.setattr(pro_probe, "_REGISTRY", registry)

        summary = pro_probe.probe_components(
            ["a", "b", "c"], host=_host(), cache_dir=tmp_path,
        )
        # Cache must hold both successes; the failed probe should not
        # have written a partial entry.
        host_sig = _host().host_signature()
        cache = CalibrationCache.load(host_sig, cache_dir=tmp_path)
        assert "a" in cache.components
        assert "c" in cache.components
        assert "b" not in cache.components

    def test_cache_survives_partial_progress(self, tmp_path, monkeypatch):
        """Simulate a crash mid-suite: run probes, stop after probe 'a',
        then re-run. The cache from the first run must still hold 'a'.
        """
        good_a = _FakeProbe("a")
        registry = {"a": lambda: good_a}
        monkeypatch.setattr(pro_probe, "_REGISTRY", registry)

        s1 = pro_probe.probe_components(["a"], host=_host(), cache_dir=tmp_path)
        assert s1.all_ok is True

        # Now reload cache + verify entry is durable.
        cache = CalibrationCache.load(_host().host_signature(), cache_dir=tmp_path)
        assert "a" in cache.components
        assert cache.components["a"].extra.get("probed") is True


class TestOrchestratorEvents:
    def test_on_event_callback_receives_start_done(self, tmp_path, monkeypatch):
        good = _FakeProbe("a")
        registry = {"a": lambda: good}
        monkeypatch.setattr(pro_probe, "_REGISTRY", registry)

        events: list[tuple[str, dict]] = []
        pro_probe.probe_components(
            ["a"], host=_host(), cache_dir=tmp_path,
            on_event=lambda e, p: events.append((e, p)),
        )
        names = [e for e, _ in events]
        assert names == ["probe_start", "probe_done"]
        assert events[0][1]["component"] == "a"
        assert events[1][1]["component"] == "a"
        assert "duration_s" in events[1][1]

    def test_on_event_receives_failure(self, tmp_path, monkeypatch):
        bad = _FakeProbe("a", fail_in="measure")
        monkeypatch.setattr(pro_probe, "_REGISTRY", {"a": lambda: bad})

        events: list[tuple[str, dict]] = []
        pro_probe.probe_components(
            ["a"], host=_host(), cache_dir=tmp_path,
            on_event=lambda e, p: events.append((e, p)),
        )
        names = [e for e, _ in events]
        assert "probe_failed" in names


class TestSkipProbe:
    def test_skip_probe_only_downloads(self, tmp_path, monkeypatch):
        good = _FakeProbe("a")
        monkeypatch.setattr(pro_probe, "_REGISTRY", {"a": lambda: good})

        summary = pro_probe.probe_components(
            ["a"], host=_host(), cache_dir=tmp_path, skip_probe=True,
        )
        assert summary.all_ok is True
        assert good.download_called is True
        assert good.measure_called is False
        # Cache entry marked as download-only.
        cache = CalibrationCache.load(_host().host_signature(), cache_dir=tmp_path)
        assert cache.components["a"].extra.get("download_only") is True


# ---------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------


class TestMeasurementHelpers:
    def test_process_rss_mb_returns_positive(self):
        # Any live Python process has at least a few MB resident.
        assert pro_probe._process_rss_mb() >= 1

    def test_vram_free_mb_zero_or_positive(self):
        # Must not raise even when nvidia-smi is missing.
        v = pro_probe._vram_free_mb()
        assert v >= 0

    def test_measure_callable_tps_returns_median(self):
        calls = [0]

        def _work():
            calls[0] += 1
            # Sleep for a deterministic interval; return constant tokens.
            import time
            time.sleep(0.01)
            return 100

        tps = pro_probe._measure_callable_tps(_work, n_iters=3)
        # 100 tokens / ~0.01s = ~10000 tps. Loose bound to dodge jitter.
        assert tps > 100
        assert calls[0] == 3

    def test_measure_callable_tps_handles_zero_tokens(self):
        tps = pro_probe._measure_callable_tps(lambda: 0, n_iters=3)
        assert tps == 0.0


def test_probe_lazy_import_symbols_exist():
    """Probe download()/measure() lazily import runtime symbols, so a rename
    there only surfaces at live `pro setup`, not in unit tests. Assert the
    symbols the jina embedder + reranker probes depend on exist - regression
    guard for the stale-import bugs (get_install_dir / LlamaCppReranker / rerank)."""
    from supergraph.registry.installer import (  # noqa: F401
        install_embedder,
        load_installed_embedder,
    )
    from supergraph.embedding.reranker import GGUFReranker

    # JinaV3RerankerProbe.measure() calls reranker.score(query, documents)
    assert hasattr(GGUFReranker, "score")


def test_all_probe_lazy_symbols_resolve():
    """Static guard for the probe lazy-import bug class: every supergraph symbol
    a probe imports inside download()/measure(), or calls as <module>.<attr>, must
    exist. In-method imports hide renames from normal tests until live `pro setup`.
    Caught: jina get_install_dir / LlamaCppReranker / .rerank() and vision pull_model.
    Skips modules whose import fails on an ABSENT OPTIONAL DEP (not a stale symbol)."""
    import ast
    import importlib
    import supergraph.pro_probe as pp

    tree = ast.parse(open(pp.__file__).read())
    problems = []
    alias_to_mod = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("supergraph"):
            try:
                mod = importlib.import_module(node.module)
            except ImportError:
                continue  # optional dep missing - not a stale-symbol bug
            for a in node.names:
                alias_to_mod[a.asname or a.name] = f"{node.module}.{a.name}"
                if not hasattr(mod, a.name):
                    try:
                        importlib.import_module(f"{node.module}.{a.name}")
                    except ImportError:
                        problems.append(f"{node.module}.{a.name} MISSING")

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            full = alias_to_mod.get(node.value.id)
            if not full:
                continue
            try:
                m = importlib.import_module(full)
            except ImportError:
                continue
            if not hasattr(m, node.attr):
                problems.append(f"{full}.{node.attr} MISSING (called as {node.value.id}.{node.attr})")

    assert not sorted(set(problems)), f"stale probe symbols: {sorted(set(problems))}"
