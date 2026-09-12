"""Tests for ONNX HF embedder. Skipped if onnxruntime/tokenizers not installed."""
import pytest
import numpy as np


class TestOnnxHFEmbedder:
    """Only runs if onnxruntime and tokenizers are installed AND model is downloaded."""

    @pytest.fixture(scope="class")
    def embedder(self):
        pytest.importorskip("onnxruntime")
        pytest.importorskip("tokenizers")
        from supergraph.registry.installer import is_installed
        if not is_installed("embeddinggemma-300m"):
            pytest.skip("embeddinggemma-300m not installed")
        from supergraph.registry.installer import load_installed_embedder
        return load_installed_embedder("embeddinggemma-300m", dims=256)

    def test_encode_queries(self, embedder):
        vecs = embedder.encode_queries(["hello world"])
        assert vecs.shape == (1, 256)
        assert vecs.dtype == np.float32

    def test_encode_documents(self, embedder):
        vecs = embedder.encode_documents(["hello world"], titles=["test"])
        assert vecs.shape == (1, 256)

    def test_dims_property(self, embedder):
        assert embedder.dims == 256


class TestRegistry:
    def test_list_models(self):
        from supergraph.registry.models import list_models
        models = list_models()
        assert len(models) >= 1
        assert models[0]["name"] == "embeddinggemma-300m"

    def test_get_model_info(self):
        from supergraph.registry.models import get_model_info
        info = get_model_info("embeddinggemma-300m")
        assert info is not None
        assert info["base_dims"] == 768
        assert "q4" in info["variants"]

    def test_unknown_model(self):
        from supergraph.registry.models import get_model_info
        assert get_model_info("nonexistent") is None


class TestInstaller:
    def test_detect_onnx_package(self):
        from supergraph.registry.installer import _detect_onnx_package
        pkg = _detect_onnx_package()
        assert pkg in ("onnxruntime", "onnxruntime-gpu")

    def test_model_dir_path(self):
        from supergraph.registry.installer import get_model_dir
        path = get_model_dir("embeddinggemma-300m")
        assert "embeddinggemma-300m" in str(path)


class _FakeSession:
    def __init__(self, providers):
        self._providers = providers

    def get_providers(self):
        return list(self._providers)


class _FakeOrt:
    def __init__(self, providers_per_call):
        self.providers_per_call = list(providers_per_call)
        self.calls = 0

    def InferenceSession(self, model_source, **kwargs):
        providers = self.providers_per_call[min(self.calls, len(self.providers_per_call) - 1)]
        self.calls += 1
        return _FakeSession(providers)


class TestSessionCreationRetry:
    def test_retries_when_cuda_requested_but_first_session_falls_back_to_cpu(self):
        from supergraph.embedding.onnx_hf_embedder import _create_inference_session

        ort = _FakeOrt([
            ["CPUExecutionProvider"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ])
        session = _create_inference_session(
            ort,
            "model.onnx",
            {"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]},
        )
        assert ort.calls == 2
        assert session.get_providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_does_not_retry_when_cuda_is_not_requested(self):
        from supergraph.embedding.onnx_hf_embedder import _create_inference_session

        ort = _FakeOrt([["CPUExecutionProvider"]])
        session = _create_inference_session(
            ort,
            "model.onnx",
            {"providers": ["CPUExecutionProvider"]},
        )
        assert ort.calls == 1
        assert session.get_providers() == ["CPUExecutionProvider"]
