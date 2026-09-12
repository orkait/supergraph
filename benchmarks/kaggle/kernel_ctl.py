#!/usr/bin/env python3
"""Kaggle kernel control via kagglesdk (KGAT bearer token auth).

Usage:
    python kernel_ctl.py status
    python kernel_ctl.py logs
    python kernel_ctl.py cancel
    python kernel_ctl.py run
    python kernel_ctl.py run --kernel supergraph-pipeline-refactored
"""

import argparse
import json
from kagglesdk import KaggleClient
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetKernelSessionStatusRequest,
    ApiCreateKernelSessionRequest,
    ApiListKernelSessionOutputRequest,
)

OWNER = "superkaiii"
DEFAULT_KERNEL = "supergraph-jina-v5-small"


def get_client():
    # Auto-loads KGAT from ~/.kaggle/access_token or KAGGLE_API_TOKEN env var
    return KaggleClient()


def status(kernel: str):
    with get_client() as c:
        req = ApiGetKernelSessionStatusRequest()
        req.user_name = OWNER
        req.kernel_slug = kernel
        resp = c.kernels.kernels_api_client.get_kernel_session_status(req)
        print(f"Status: {resp.status}")
        if resp.failure_message:
            print(f"Failure: {resp.failure_message}")
        return resp.status


def logs(kernel: str):
    with get_client() as c:
        req = ApiListKernelSessionOutputRequest()
        req.user_name = OWNER
        req.kernel_slug = kernel
        resp = c.kernels.kernels_api_client.list_kernel_session_output(req)

        if not resp.log:
            print("No logs available yet.")
            return

        log_data = json.loads(resp.log)
        for entry in log_data:
            stream = entry.get("stream_name", "")
            data = entry.get("data", "").rstrip("\n")
            prefix = "ERR" if stream == "stderr" else "OUT"
            print(f"[{prefix}] {data}")


def cancel(kernel: str):
    # Note: cancel_kernel_session requires kernel_session_id from a prior run() response.
    # Workaround: use kaggle CLI instead
    import subprocess
    try:
        subprocess.check_call(["kaggle", "kernels", "status", f"{OWNER}/{kernel}"])
        print("[INFO] Use kaggle CLI to cancel: kaggle kernels push --kernel-metadata benchmarks/kaggle/kernel-metadata.json")
        return
    except:
        pass


def run(kernel: str):
    with get_client() as c:
        req = ApiCreateKernelSessionRequest()
        req.slug = f"{OWNER}/{kernel}"
        req.kernel_type = "SCRIPT"
        req.language = "PYTHON"
        req.machine_shape = "GPU_P100_X4"
        req.enable_internet = True
        resp = c.kernels.kernels_api_client.create_kernel_session(req)
        print(f"Started: {resp}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["status", "logs", "cancel", "run"])
    p.add_argument("--kernel", default=DEFAULT_KERNEL)
    args = p.parse_args()

    if args.command == "status":
        status(args.kernel)
    elif args.command == "logs":
        logs(args.kernel)
    elif args.command == "cancel":
        cancel(args.kernel)
    elif args.command == "run":
        run(args.kernel)
