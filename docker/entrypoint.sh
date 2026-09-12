#!/usr/bin/env sh
# supergraph docker entrypoint.
#
# The playground refuses to bind 0.0.0.0 without auth - good. To make
# the container "just start", we auto-generate a random token on first
# boot and write it to /data/.auth_token so it persists across
# restarts (volume-mounted /data is the default). Operators who want a
# fixed token set SUPERGRAPH_AUTH_TOKEN themselves; operators who
# accept a private-network deployment without auth set
# SUPERGRAPH_ALLOW_UNAUTH_BIND=1.
set -eu

DB="${SUPERGRAPH_DB_PATH:-/data}"
HOST="${SUPERGRAPH_HOST:-0.0.0.0}"
# SUPERGRAPH_PORT wins; else a platform-injected PORT (Railway/Heroku/Fly); else 7200
PORT="${SUPERGRAPH_PORT:-${PORT:-7200}}"
TOKEN_FILE="${DB}/.auth_token"

# When CUDA libs ship as pip wheels (Pro image), supergraph.gpu.setup()
# must run BEFORE any import of llama_cpp / onnxruntime - it ctypes-
# preloads libcudart / libcublas / libcudnn from the wheels via
# RTLD_GLOBAL so subsequent dlopens resolve symbols correctly. Skipping
# this on hosts without the wheels is harmless (setup() returns ready=False
# and the call is a no-op for shared libs already on the system).
if [ "${SUPERGRAPH_GPU:-0}" = "1" ]; then
    python -c "from supergraph import gpu; s = gpu.setup(); print('[entrypoint] gpu.setup ready=' + str(s.ready) + ' provider=' + str(s.provider) + ' err=' + str(s.error or ''))" || true
fi

if [ -z "${SUPERGRAPH_AUTH_TOKEN:-}" ] && [ "${SUPERGRAPH_ALLOW_UNAUTH_BIND:-0}" != "1" ]; then
    mkdir -p "${DB}"
    if [ ! -s "${TOKEN_FILE}" ]; then
        python -c "import secrets; print(secrets.token_urlsafe(32))" > "${TOKEN_FILE}"
        chmod 600 "${TOKEN_FILE}" || true
        echo "[entrypoint] generated new auth token at ${TOKEN_FILE}"
    fi
    SUPERGRAPH_AUTH_TOKEN="$(cat "${TOKEN_FILE}")"
    export SUPERGRAPH_AUTH_TOKEN
    echo "[entrypoint] auth token (use as 'Authorization: Bearer <token>'): ${SUPERGRAPH_AUTH_TOKEN}"
fi

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

exec supergraph playground \
    --host "${HOST}" \
    --port "${PORT}" \
    --no-browser \
    --db-path "${DB}"
