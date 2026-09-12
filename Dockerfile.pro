# supergraph - Pro GPU image (slim).
#
# Single-stage on python:3.12-slim-bookworm. CUDA 12.4 runtime is
# delivered via the official nvidia-* pip wheels (cuda-runtime, cublas,
# cudnn, nvrtc, nvjitlink). Slim base + pip-delivered CUDA produces a
# ~4.5 GB image vs ~6 GB for the nvidia/cuda:cudnn-runtime base.
#
# CRITICAL: install + cleanup + strip + purge run in a SINGLE RUN.
# Splitting into separate layers caused ~2 GB of duplicated unstripped
# .so files in v3 (strip in a later layer added stripped copies on top
# of the originals). Keep all binary mutations in the same layer.
#
# Pre-downloads every model the default ProSpec selects (Bonsai TQ1_0
# GGUF, TinyBERT NER ONNX, Jina v5 small ONNX). Skip with --build-arg
# SKIP_MODEL_PREFETCH=1 for a ~3.5 GB image (first start downloads).
#
# Calibration is intentionally NOT run at build time - calibration is
# host-specific (per-CPU/per-GPU TPS measurements) and a value measured
# on a CI runner means nothing on the deployment host. Run it once on
# the target machine inside the container:
#
#     docker compose --profile pro run --rm supergraph-pro supergraph pro setup
#
# The calibration cache lives on the gs-cache volume and persists across
# container restarts.
#
# Build:  docker buildx build -f Dockerfile.pro --builder supergraph-builder \
#             -t supergraph-pro:latest --load .
# Run:    docker run --gpus all --cpus=8 --memory=16g -p 7200:7200 \
#               -v gs-data:/data -v gs-cache:/root/.cache/supergraph \
#               supergraph-pro:latest

ARG SUPERGRAPH_VERSION=0.6.0
ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE}

ARG SUPERGRAPH_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SUPERGRAPH_DB_PATH=/data \
    SUPERGRAPH_HOST=0.0.0.0 \
    SUPERGRAPH_PORT=7200 \
    HF_HOME=/root/.cache/huggingface

# Bring in uv as a build-only binary (deleted at end of mega-RUN).
COPY --from=ghcr.io/astral-sh/uv:0.5.14 /uv /uvx /usr/local/bin/

# Mega-RUN: install all wheels + CUDA runtime + supergraph + strip
# symbols + delete unused ORT execution providers + purge build deps.
# Everything lands in one layer so cleanup actually shrinks the image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates binutils libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && uv pip install --system \
        nvidia-cuda-runtime-cu12 \
        nvidia-cublas-cu12 \
        nvidia-cudnn-cu12 \
        nvidia-cuda-nvrtc-cu12 \
        nvidia-nvjitlink-cu12 \
    && uv pip install --system \
        --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/cu124" \
        "llama-cpp-python>=0.3" \
    && uv pip install --system "onnxruntime-gpu>=1.17" \
    && uv pip install --system \
        "supergraph==${SUPERGRAPH_VERSION}" \
        "tokenizers>=0.15" \
        "huggingface-hub>=0.24" \
        "mcp>=1.0" \
    && find /usr/local/lib/python3.12/site-packages -depth \
        \( -type d \( -name '__pycache__' -o -name 'tests' -o -name 'test' \) \
        -o -name '*.pyi' -o -name '*.pyc' \) \
        -exec rm -rf {} + 2>/dev/null \
    && find /usr/local/lib/python3.12/site-packages -depth \
        \( -name '*.md' -o -name '*.rst' -o -name 'AUTHORS*' \
        -o -name 'CHANGELOG*' -o -name 'NOTICE*' \) \
        -delete 2>/dev/null \
    && cd /usr/local/lib/python3.12/site-packages/onnxruntime/capi \
    && rm -f libonnxruntime_providers_tensorrt* \
             libonnxruntime_providers_qnn* \
             libonnxruntime_providers_dml* \
             libonnxruntime_providers_openvino* \
             libonnxruntime_providers_migraphx* \
             libonnxruntime_providers_cann* \
    && cd / \
    && find /usr/local/lib/python3.12/site-packages/nvidia \
            -name '*.so*' -type f \
            -exec strip --strip-unneeded {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.12/site-packages/llama_cpp/lib \
            -name '*.so*' -type f \
            -exec strip --strip-unneeded {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.12/site-packages/onnxruntime \
            -name '*.so*' -type f \
            -exec strip --strip-unneeded {} + 2>/dev/null || true \
    && apt-get purge -y --auto-remove binutils \
    && rm -rf /var/lib/apt/lists/* \
              /usr/local/bin/uv /usr/local/bin/uvx \
              /root/.cache /tmp/*

# Pre-pull every model the default ProSpec selects so a cold container
# is functional immediately. Public repos only - no HF_TOKEN required at
# build time. ~1.2 GB total (Bonsai TQ1_0 GGUF ~1 GB + Jina v5 ONNX
# ~120 MB + TinyBERT NER ONNX). Skip with --build-arg SKIP_MODEL_PREFETCH=1
# for a ~3.5 GB image; first start downloads on demand.
ARG SKIP_MODEL_PREFETCH=0
RUN if [ "$SKIP_MODEL_PREFETCH" = "0" ]; then \
        python -c "from huggingface_hub import hf_hub_download; \
            hf_hub_download('superkaiii/Ternary-Bonsai-4B-GGUF', \
                            'Ternary-Bonsai-4B-TQ1_0.gguf')" \
        && python -c "from huggingface_hub import snapshot_download; \
            snapshot_download('jinaai/jina-embeddings-v3', \
                              allow_patterns=['*.onnx','*.json','*.txt'])" \
        && python -c "from huggingface_hub import snapshot_download; \
            snapshot_download('onnx-community/TinyBERT-finetuned-NER-ONNX')" ; \
    fi

COPY docker/entrypoint.sh /usr/local/bin/supergraph-entrypoint
RUN chmod +x /usr/local/bin/supergraph-entrypoint \
    && mkdir -p /data /root/.cache/supergraph

WORKDIR /root
VOLUME ["/data", "/root/.cache/supergraph", "/root/.cache/huggingface"]

EXPOSE 7200

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(3); \
        s.connect(('127.0.0.1', 7200)); s.close()" || exit 1

ENTRYPOINT ["/usr/local/bin/supergraph-entrypoint"]
