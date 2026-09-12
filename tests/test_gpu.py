from __future__ import annotations

from unittest.mock import patch

import pytest

from supergraph import gpu


@pytest.fixture(autouse=True)
def _reset_gpu_status(monkeypatch):
    snapshot = dict(__import__("os").environ)
    gpu.reset_for_tests()
    yield
    gpu.reset_for_tests()
    import os
    for key in list(os.environ.keys()):
        if key not in snapshot:
            del os.environ[key]
        elif os.environ[key] != snapshot[key]:
            os.environ[key] = snapshot[key]


class TestStatusCaching:
    def test_is_ready_false_before_setup(self):
        assert gpu.is_ready() is False
        assert gpu.status() is None

    def test_n_gpu_layers_default_zero_before_setup(self):
        assert gpu.n_gpu_layers_default() == 0

    def test_setup_caches_result(self):
        with patch.object(gpu, "_find_nvidia_libs", return_value=[]), \
             patch.object(gpu, "_probe_onnxruntime",
                          return_value=(False, None, "test: not installed")), \
             patch.object(gpu, "_probe_llama_cpp",
                          return_value=(False, None, "test: not installed")):
            s1 = gpu.setup()
            s2 = gpu.setup()
        assert s1 is s2


class TestProbeFailure:
    def test_setup_no_providers_marks_not_ready(self):
        with patch.object(gpu, "_find_nvidia_libs", return_value=[]), \
             patch.object(gpu, "_probe_onnxruntime",
                          return_value=(False, None, "ort missing")), \
             patch.object(gpu, "_probe_llama_cpp",
                          return_value=(False, None, "llama-cpp missing")):
            s = gpu.setup()
        assert s.ready is False
        assert s.error is not None
        assert "ort missing" in s.error
        assert "llama-cpp missing" in s.error
        assert gpu.n_gpu_layers_default() == 0

    def test_either_probe_passing_marks_ready(self):
        with patch.object(gpu, "_find_nvidia_libs", return_value=[]), \
             patch.object(gpu, "_probe_onnxruntime",
                          return_value=(True, "CUDAExecutionProvider", None)), \
             patch.object(gpu, "_probe_llama_cpp",
                          return_value=(False, None, "wheel CPU-only")), \
             patch.object(gpu, "_read_device_name", return_value="MockGPU"):
            s = gpu.setup()
        assert s.ready is True
        assert s.provider == "CUDAExecutionProvider"
        assert s.device_name == "MockGPU"
        assert gpu.is_ready() is True
        assert gpu.n_gpu_layers_default() == -1


class TestLibDiscovery:
    def test_find_returns_empty_when_no_nvidia_dir(self, tmp_path, monkeypatch):
        site = tmp_path / "site-packages"
        site.mkdir()
        monkeypatch.setattr("sys.path", [str(site)])
        assert gpu._find_nvidia_libs() == []

    def test_find_returns_libs_in_load_order(self, tmp_path, monkeypatch):
        site = tmp_path / "site-packages"
        site.mkdir()
        for comp, soname in [("cudnn", "libcudnn.so.9"),
                             ("cuda_runtime", "libcudart.so.12")]:
            d = site / "nvidia" / comp / "lib"
            d.mkdir(parents=True)
            (d / soname).write_bytes(b"")
        monkeypatch.setattr("sys.path", [str(site)])
        libs = gpu._find_nvidia_libs()
        names = [p.name for p in libs]
        assert names.index("libcudart.so.12") < names.index("libcudnn.so.9")


class TestSetupSurfacesEnvFlag:
    def test_setup_ready_sets_supergraph_gpu(self, monkeypatch):
        monkeypatch.delenv("SUPERGRAPH_GPU", raising=False)
        with patch.object(gpu, "_find_nvidia_libs", return_value=[]), \
             patch.object(gpu, "_probe_onnxruntime",
                          return_value=(True, "CUDAExecutionProvider", None)), \
             patch.object(gpu, "_probe_llama_cpp",
                          return_value=(True, "llama_cpp_cuda", None)), \
             patch.object(gpu, "_read_device_name", return_value=None):
            gpu.setup()
        import os
        assert os.environ.get("SUPERGRAPH_GPU") == "1"

    def test_setup_failed_does_not_touch_env(self, monkeypatch):
        monkeypatch.delenv("SUPERGRAPH_GPU", raising=False)
        with patch.object(gpu, "_find_nvidia_libs", return_value=[]), \
             patch.object(gpu, "_probe_onnxruntime",
                          return_value=(False, None, "no ort")), \
             patch.object(gpu, "_probe_llama_cpp",
                          return_value=(False, None, "no llama-cpp")):
            gpu.setup()
        import os
        assert "SUPERGRAPH_GPU" not in os.environ


class TestPreloadBestEffort:
    def test_preload_skips_unloadable(self, tmp_path):
        bogus = tmp_path / "libnope.so.999"
        bogus.write_bytes(b"\x00\x00not a real elf")
        loaded = gpu._preload([bogus])
        assert "libnope.so.999" not in loaded
