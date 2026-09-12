
import ctypes
import os
import sys
from pathlib import Path

import numpy as np

from supergraph.embedding.base import Embedder
from supergraph.embedding.postprocess import l2_normalize, truncate_dims


_CU12_PRELOADED = False

_CU12_SONAMES = (
    "libcudart.so.12",
    "libnvrtc.so.12",
    "libnvJitLink.so.12",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcufft.so.11",
    "libcurand.so.10",
    "libcudnn.so.9",
)

_WHEEL_LAYOUT: dict[str, str] = {
    "libcudart.so.12": "cuda_runtime/lib",
    "libnvrtc.so.12": "cuda_nvrtc/lib",
    "libnvJitLink.so.12": "nvjitlink/lib",
    "libcublas.so.12": "cublas/lib",
    "libcublasLt.so.12": "cublas/lib",
    "libcufft.so.11": "cufft/lib",
    "libcurand.so.10": "curand/lib",
    "libcudnn.so.9": "cudnn/lib",
}


def _userns_restrict_active() -> tuple[bool, str | None]:
    if os.path.exists("/.dockerenv"):
        return False, None
    path = "/proc/sys/kernel/apparmor_restrict_unprivileged_userns"
    try:
        with open(path) as f:
            raw = f.read().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return False, None
    return raw == "1", raw


def _system_cuda_libs_present() -> bool:
    search: list[str] = []
    for env in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(env)
        if value:
            search.append(os.path.join(value, "lib64"))
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    search.extend(p for p in ld.split(":") if p)
    search.extend([
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ])
    for soname in _CU12_SONAMES:
        if not any(os.path.exists(os.path.join(d, soname)) for d in search):
            return False
    return True


def _preload_cu12_libs() -> None:
    global _CU12_PRELOADED
    if _CU12_PRELOADED or sys.platform != "linux":
        return

    userns_blocked, userns_raw = _userns_restrict_active()
    if userns_blocked:
        raise RuntimeError(
            "CUDA initialization is likely blocked on this host:\n"
            "  /proc/sys/kernel/apparmor_restrict_unprivileged_userns "
            f"= {userns_raw}\n"
            "  This sysctl restricts unprivileged user namespaces, which the\n"
            "  CUDA runtime requires to initialize from a venv-isolated\n"
            "  process. cudaGetDeviceCount() will fail with error 304.\n"
            "\n"
            "Verify the symptom (returns 0 = blocked, 1 = allowed):\n"
            "    cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns\n"
            "\n"
            "Workarounds (root, distro-dependent):\n"
            "  - Allow temporarily for this boot:\n"
            "      sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0\n"
            "  - Or persist via /etc/sysctl.d/, or run inside a container.\n"
            "  - Restore later:\n"
            "      sudo sysctl kernel.apparmor_restrict_unprivileged_userns=1\n"
            "\n"
            "If the symptom does not match what you observe, please report\n"
            "with `nvidia-smi` output + this message at "
            "https://github.com/orkait/supergraph/issues."
        )

    if _system_cuda_libs_present():
        _CU12_PRELOADED = True
        return

    site_packages_dirs = [p for p in sys.path if "site-packages" in p]
    added_paths = []
    for sp in site_packages_dirs:
        nvidia_root = Path(sp) / "nvidia"
        if not nvidia_root.exists():
            continue
        for soname in _CU12_SONAMES:
            lib_dir = nvidia_root / _WHEEL_LAYOUT[soname]
            if lib_dir.exists():
                added_paths.append(str(lib_dir))
                lib_path = lib_dir / soname
                if lib_path.exists():
                    try:
                        ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass
    
    if added_paths:
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld = ":".join(added_paths)
        if current_ld:
            new_ld = f"{new_ld}:{current_ld}"
        os.environ["LD_LIBRARY_PATH"] = new_ld

    _CU12_PRELOADED = True


def _resolve_providers(providers: list[str] | str | None) -> list[str]:
    wanted: list[str] = []
    if providers is not None:
        if isinstance(providers, str):
            wanted = [p.strip() for p in providers.split(",") if p.strip()]
        else:
            wanted = list(providers)
    else:
        env_providers = os.environ.get("SUPERGRAPH_ORT_PROVIDERS")
        if env_providers:
            wanted = [p.strip() for p in env_providers.split(",") if p.strip()]
        elif os.environ.get("SUPERGRAPH_GPU") == "1":
            wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            wanted = ["CPUExecutionProvider"]

    if any(p in ("CUDAExecutionProvider", "TensorrtExecutionProvider") for p in wanted):
        _preload_cu12_libs()

    import onnxruntime as ort
    ort.get_available_providers()

    resolved = list(wanted)
    
    if "CPUExecutionProvider" not in resolved:
        resolved.append("CPUExecutionProvider")
    return resolved


def _cuda_requested(providers: list[str]) -> bool:
    return any(p == "CUDAExecutionProvider" for p in providers)


def _create_inference_session(ort, model_source, sess_kwargs: dict):
    session = ort.InferenceSession(model_source, **sess_kwargs)
    requested = sess_kwargs.get("providers", [])
    if any(p in ("CUDAExecutionProvider", "TensorrtExecutionProvider") for p in requested):
        active = list(session.get_providers())
        if not any(p in ("CUDAExecutionProvider", "TensorrtExecutionProvider") for p in active):
            session = ort.InferenceSession(model_source, **sess_kwargs)
            active = list(session.get_providers())
            if not any(p in ("CUDAExecutionProvider", "TensorrtExecutionProvider") for p in active):
                raise RuntimeError(
                    f"GPU provider requested but unavailable. Active: {active}. "
                    "Check LD_LIBRARY_PATH and nvidia-* wheels."
                )
    return session


def _patch_gqa_for_cuda(onnx_path: str | Path) -> bytes | None:
    try:
        import onnx
    except ImportError:
        import logging
        logging.getLogger(__name__).warning(
            "onnx package not installed - cannot patch GroupQueryAttention "
            "for CUDA. Decoder models may fail on GPU. "
            "Fix: pip install onnx"
        )
        return None

    model = onnx.load(str(onnx_path))
    patched = False
    for node in model.graph.node:
        if node.op_type == "GroupQueryAttention" and len(node.input) > 9:
            while len(node.input) > 9:
                node.input.pop()
            patched = True

    if not patched:
        return None
    return model.SerializeToString()


class OnnxHFEmbedder(Embedder):

    def __init__(
        self,
        model_dir: str | Path,
        output_dims: int | None = None,
        query_prefix: str = "",
        doc_prefix_template: str = "",
        max_length: int = 512,
        pooling_mode: str = "mean",
        onnx_file: str | None = None,
        providers: list[str] | str | None = None,
        gpu_mem_limit: int | None = None,
    ):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "OnnxHFEmbedder requires onnxruntime and tokenizers. "
                "Install with: supergraph install-embedder embeddinggemma"
            ) from e

        if pooling_mode not in ("mean", "last_token"):
            raise ValueError(
                f"pooling_mode must be 'mean' or 'last_token', got {pooling_mode!r}"
            )

        model_dir = Path(model_dir)
        self._output_dims = output_dims
        self._query_prefix = query_prefix
        self._doc_prefix_template = doc_prefix_template
        self._max_length = max_length
        self._pooling_mode = pooling_mode

        tok_path = model_dir / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")
        self._tokenizer = Tokenizer.from_file(str(tok_path))
        self._max_length = min(max_length, 512)
        self._tokenizer.enable_padding(pad_id=0, pad_to_multiple_of=128)
        self._tokenizer.enable_truncation(max_length=self._max_length)

        if onnx_file:
            candidate = model_dir / onnx_file
            if not candidate.exists():
                raise FileNotFoundError(f"ONNX file not found: {candidate}")
            onnx_path = candidate
        else:
            onnx_files = list(model_dir.glob("*.onnx"))
            if not onnx_files:
                onnx_dir = model_dir / "onnx"
                if onnx_dir.exists():
                    onnx_files = list(onnx_dir.glob("*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(f"No .onnx file found in {model_dir}")
            onnx_path = onnx_files[0]

        self._providers = _resolve_providers(providers)
        uses_gpu = any(
            p in self._providers
            for p in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
        )

        provider_options = None
        if gpu_mem_limit and uses_gpu:
            limit_bytes = int(gpu_mem_limit) * 1024 * 1024 * 1024
            provider_options = [
                {"gpu_mem_limit": str(limit_bytes)}
                if p == "CUDAExecutionProvider" else {}
                for p in self._providers
            ]

        from supergraph.core.compute_profile import get_profile
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if uses_gpu:
            sess_options.add_session_config_entry("session.disable_matmul_nbits", "1")
        else:
            sess_options.intra_op_num_threads = get_profile().embed_threads
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        sess_kwargs: dict = {
            "sess_options": sess_options,
            "providers": self._providers,
        }
        if provider_options:
            sess_kwargs["provider_options"] = provider_options

        model_bytes = _patch_gqa_for_cuda(onnx_path) if uses_gpu else None
        if model_bytes:
            self._session = _create_inference_session(ort, model_bytes, sess_kwargs)
        else:
            self._session = _create_inference_session(ort, str(onnx_path), sess_kwargs)
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._needs_token_type_ids = "token_type_ids" in self._input_names
        self._needs_position_ids = "position_ids" in self._input_names

        self._kv_cache_specs: list = []
        for inp in self._session.get_inputs():
            if inp.name.startswith("past_key_values."):
                shape = inp.shape
                num_heads = shape[1] if isinstance(shape[1], int) else None
                head_dim = shape[3] if isinstance(shape[3], int) else None
                dtype = np.float16 if inp.type == "tensor(float16)" else np.float32
                self._kv_cache_specs.append((inp.name, num_heads, head_dim, dtype))

        outputs = self._session.get_outputs()
        self._hidden_output_idx = 0
        for i, out in enumerate(outputs):
            if out.name == "sentence_embedding":
                self._hidden_output_idx = i
                break
        else:
            for i, out in enumerate(outputs):
                if out.name == "last_hidden_state" or "hidden" in out.name:
                    self._hidden_output_idx = i
                    break

        test_enc = self._tokenizer.encode("test")
        test_ids = np.array([[test_enc.ids[0]]], dtype=np.int64)
        test_mask = np.array([[1]], dtype=np.int64)
        test_feed = {"input_ids": test_ids, "attention_mask": test_mask}
        if self._needs_token_type_ids:
            test_feed["token_type_ids"] = np.zeros((1, 1), dtype=np.int64)
        if self._needs_position_ids:
            test_feed["position_ids"] = np.zeros((1, 1), dtype=np.int64)
        for name, num_heads, head_dim, dtype in self._kv_cache_specs:
            test_feed[name] = np.zeros((1, num_heads, 0, head_dim), dtype=dtype)
        test_out = self._session.run(None, test_feed)
        self._base_dims = test_out[self._hidden_output_idx].shape[-1]
        if self._output_dims is None:
            self._output_dims = self._base_dims

    @property
    def name(self) -> str:
        return "onnx-hf"

    @property
    def dims(self) -> int:
        return self._output_dims

    def encode_documents(self, texts: list[str], titles: list[str | None] | None = None) -> np.ndarray:
        if not texts:
            return np.empty((0, self._output_dims), dtype=np.float32)
        prefixed = []
        for i, text in enumerate(texts):
            title = titles[i] if titles and i < len(titles) and titles[i] else "none"
            prefix = self._doc_prefix_template.format(title=title) if self._doc_prefix_template else ""
            prefixed.append(f"{prefix}{text}")
        return self._encode(prefixed)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._output_dims), dtype=np.float32)
        prefixed = [f"{self._query_prefix}{t}" for t in texts]
        return self._encode(prefixed)

    def _encode(self, texts: list[str]) -> np.ndarray:
        safe_texts = [t if (t and not t.isspace()) else " " for t in texts]

        encoded = self._tokenizer.encode_batch(safe_texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        embeddings = self._encode_feed_dict(input_ids, attention_mask)

        if embeddings.ndim == 2:
            pooled = embeddings
        elif self._pooling_mode == "last_token":
            last_idx = attention_mask.sum(axis=1) - 1
            batch_idx = np.arange(embeddings.shape[0])
            pooled = embeddings[batch_idx, last_idx]
        else:
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            pooled = (embeddings * mask_expanded).sum(axis=1) / mask_expanded.sum(axis=1)

        if self._output_dims < pooled.shape[1]:
            pooled = truncate_dims(pooled, self._output_dims)
        else:
            pooled = l2_normalize(pooled)

        return pooled.astype(np.float32)

    def _encode_feed_dict(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self._needs_token_type_ids:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        if self._needs_position_ids:
            pos = attention_mask.cumsum(axis=1) - 1
            feed["position_ids"] = np.maximum(pos, 0)
        if self._kv_cache_specs:
            for name, num_heads, head_dim, dtype in self._kv_cache_specs:
                feed[name] = np.zeros((input_ids.shape[0], num_heads, 0, head_dim), dtype=dtype)
        outputs = self._session.run(None, feed)
        return outputs[self._hidden_output_idx]
