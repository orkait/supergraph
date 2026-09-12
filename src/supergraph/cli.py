
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path


def _open_browser(url: str) -> None:
    import time

    time.sleep(1)
    webbrowser.open(url)


def cmd_install_embedder(args: argparse.Namespace) -> None:
    from supergraph.registry.installer import install_embedder

    try:
        install_embedder(args.name, variant=args.variant)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


def cmd_list_embedders(args: argparse.Namespace) -> None:
    from supergraph.registry.models import list_models
    from supergraph.registry.installer import is_installed

    models = list_models()
    print(f"{'NAME':<30} {'STATUS':<12} {'DIMS':<8} DESCRIPTION")
    print("-" * 80)
    for m in models:
        status = "installed" if is_installed(m["name"]) else "available"
        print(f"{m['name']:<30} {status:<12} {m['base_dims']:<8} {m['description']}")


def cmd_uninstall_embedder(args: argparse.Namespace) -> None:
    from supergraph.registry.installer import uninstall_embedder

    uninstall_embedder(args.name)


def _is_loopback_host(host: str) -> bool:
    if not host:
        return False
    lo = host.strip().lower()
    if lo in ("127.0.0.1", "localhost", "::1", "[::1]"):
        return True
    return False


def cmd_playground(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        print(
            "Missing dependencies. Install with:\n"
            "  pip install supergraph[playground]",
            file=sys.stderr,
        )
        sys.exit(1)

    import os

    auth_token_set = bool(os.environ.get("SUPERGRAPH_AUTH_TOKEN"))
    allow_unauth = os.environ.get("SUPERGRAPH_ALLOW_UNAUTH_BIND") == "1"
    if not _is_loopback_host(args.host) and not auth_token_set and not allow_unauth:
        print(
            f"Refusing to bind playground to {args.host!r} without authentication.\n"
            "\n"
            "The playground accepts arbitrary DSL including VAULT READ, INGEST,\n"
            "and SYS commands. Exposing it to a non-loopback address without an\n"
            "auth token lets anyone on the network execute queries against your\n"
            "supergraph.\n"
            "\n"
            "Fix: set an auth token before starting\n"
            "    export SUPERGRAPH_AUTH_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')\n"
            "    supergraph playground --host 0.0.0.0\n"
            "\n"
            "Clients must then send ``Authorization: Bearer <token>`` on\n"
            "every /api/* request.\n"
            "\n"
            "If you really need to disable this check (e.g. inside a private\n"
            "network), set SUPERGRAPH_ALLOW_UNAUTH_BIND=1.",
            file=sys.stderr,
        )
        sys.exit(2)

    from supergraph.server import app, mount_static

    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent / "apps" / "playground" / "dist",
        here.parent.parent / "playground" / "dist",
        here.parent / "playground_dist",
    ]
    for d in candidates:
        if d.is_dir():
            mount_static(app, d)
            break

    if not args.no_browser:
        url = f"http://{args.host}:{args.port}"
        threading.Thread(target=_open_browser, args=(url,), daemon=True).start()

    if args.db_path:
        os.environ["SUPERGRAPH_DB_PATH"] = args.db_path

    uvicorn.run(app, host=args.host, port=args.port)


def cmd_vision(args: argparse.Namespace) -> None:
    try:
        from supergraph.ingest import vision_sidecar as vs
    except ImportError:
        print(
            "Missing dependencies. Install with:\n"
            "  pip install 'supergraph[vision]'",
            file=sys.stderr,
        )
        sys.exit(1)

    sub = args.vision_command
    if sub == "serve":
        repo = getattr(args, "repo", None)
        model_file = getattr(args, "model_file", None)
        mmproj_file = getattr(args, "mmproj_file", None)
        chat_format = getattr(args, "chat_format", None)
        if repo or model_file or mmproj_file or chat_format:
            if not (repo and model_file and mmproj_file):
                print(
                    "--repo, --model-file, --mmproj-file must all be set "
                    "when overriding the preset",
                    file=sys.stderr,
                )
                sys.exit(2)
            spec = vs.VLMModelSpec(
                repo=repo,
                model_file=model_file,
                mmproj_file=mmproj_file,
                chat_format=chat_format or "llava-1-5",
            )
        else:
            spec = getattr(args, "model", None)
        if args.pull_only:
            model_path, mmproj_path = vs.download_weights(spec)
            print(f"model:  {model_path}")
            print(f"mmproj: {mmproj_path}")
            return
        try:
            st = vs.start(
                host=getattr(args, "host", None),
                port=getattr(args, "port", None),
                model=spec,
                n_threads=args.threads,
                n_ctx=args.n_ctx,
            )
        except (RuntimeError, TimeoutError) as e:
            print(f"sidecar failed to start: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"vision sidecar running at {st.base_url} (pid={st.pid}, model={st.model})")
    elif sub == "stop":
        ok = vs.stop()
        print("stopped" if ok else "no sidecar was running")
    elif sub == "status":
        st = vs.status(host=getattr(args, "host", None))
        if st.running:
            print(f"running  pid={st.pid}  port={st.port}  model={st.model}  url={st.base_url}")
        else:
            print("not running")
    elif sub == "logs":
        log = vs._log_file()
        if not log.exists():
            print("no logs yet", file=sys.stderr)
            sys.exit(1)
        with log.open("rb") as f:
            sys.stdout.buffer.write(f.read())
    elif sub == "models":
        print(f"{'NAME':<18} {'DISK':<8} REPO / FILE")
        print("-" * 80)
        for name, s in vs.VLM_MODELS.items():
            size = f"{s.disk_mb} MB" if s.disk_mb else "-"
            print(f"{name:<18} {size:<8} {s.repo}/{s.model_file}")
    else:
        print("unknown vision subcommand", file=sys.stderr)
        sys.exit(2)


def cmd_config(args: argparse.Namespace) -> None:
    import json
    import msgspec
    from supergraph.config import SuperGraphConfig, load_config

    if args.schema:
        schema = msgspec.json.schema(SuperGraphConfig)
        print(json.dumps(schema, indent=2))
    elif args.defaults:
        data = msgspec.json.decode(msgspec.json.encode(SuperGraphConfig()))
        print(json.dumps(data, indent=2))
    elif args.path:
        config = load_config(args.path)
        data = msgspec.json.decode(msgspec.json.encode(config))
        print(json.dumps(data, indent=2))
    else:
        data = msgspec.json.decode(msgspec.json.encode(SuperGraphConfig()))
        print(json.dumps(data, indent=2))


def cmd_pro(args: argparse.Namespace) -> None:
    import json
    from supergraph import pro

    sub = args.pro_command
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None

    spec_kwargs: dict = {}
    for slot in ("embedder", "reranker", "ingest_mode", "bonsai_quant",
                 "bonsai_skill", "vision", "audio", "ner"):
        val = getattr(args, slot, None)
        if val is not None:
            spec_kwargs[slot] = val
    spec = pro.ProSpec(**spec_kwargs)

    if sub in ("check", "status"):
        host = pro.HostSnapshot.capture(cache_dir=cache_dir, probe_gpu=True)
        try:
            pro.check_extras_installed(spec, host)
        except pro.ProExtraNotInstalled as e:
            if args.json:
                print(json.dumps({
                    "fits": False,
                    "error": "extra_not_installed",
                    "missing_dists": e.missing_dists,
                }, indent=2))
            else:
                print(f"[pro] {e}", file=sys.stderr)
            sys.exit(2)

        rc = pro.resolve(spec, host=host, cache_dir=cache_dir)
        if args.json:
            print(json.dumps(_pro_resolved_to_json(rc), indent=2, default=str))
        else:
            _print_pro_resolved(rc, host)
        sys.exit(0 if rc.fits else (3 if rc.calibration_source == "missing" else 1))

    if sub in ("setup", "probe"):
        from supergraph import pro_probe
        host = pro_probe.HostSnapshot.capture(cache_dir=cache_dir)
        try:
            pro.check_extras_installed(spec, host)
        except pro.ProExtraNotInstalled as e:
            if args.json:
                print(json.dumps({
                    "fits": False,
                    "error": "extra_not_installed",
                    "missing_dists": e.missing_dists,
                }, indent=2))
            else:
                print(f"[pro] {e}", file=sys.stderr)
            sys.exit(2)

        component_ids = spec.component_ids()
        events: list[dict] = []

        def _emit(event: str, payload: dict) -> None:
            payload["event"] = event
            events.append(payload)
            if not args.json:
                if event == "probe_start":
                    print(f"[pro] probing {payload['component']} ...",
                          flush=True)
                elif event == "probe_done":
                    print(f"[pro] OK   {payload['component']} "
                          f"({payload['duration_s']:.1f}s)", flush=True)
                elif event == "probe_failed":
                    print(f"[pro] FAIL {payload['component']}: "
                          f"{payload['error']}", flush=True)

        skip_probe = False
        summary = pro_probe.probe_components(
            component_ids,
            host=host,
            cache_dir=cache_dir,
            on_event=_emit,
            skip_probe=skip_probe,
        )

        if args.json:
            print(json.dumps({
                "command": sub,
                "all_ok": summary.all_ok,
                "successes": [r.component_id for r in summary.successes],
                "failures": [
                    {"component": r.component_id, "error": r.error}
                    for r in summary.failures
                ],
                "total_duration_s": summary.total_duration_s,
                "events": events,
            }, indent=2))
        else:
            print()
            print(f"[pro] done in {summary.total_duration_s:.1f}s; "
                  f"{len(summary.successes)} ok, "
                  f"{len(summary.failures)} failed.")
            if summary.failures:
                print("Failures:")
                for r in summary.failures:
                    print(f"  - {r.component_id}: {r.error}")
        sys.exit(0 if summary.all_ok else 1)

    print(f"unknown pro subcommand: {sub!r}", file=sys.stderr)
    sys.exit(2)


def _pro_resolved_to_json(rc) -> dict:
    return {
        "fits": rc.fits,
        "spec": {f: getattr(rc.spec, f) for f in rc.spec.__struct_fields__},
        "host": {
            "ram_total_mb": rc.host.ram_total_mb,
            "ram_available_mb": rc.host.ram_available_mb,
            "disk_free_mb": rc.host.disk_free_mb,
            "cpu_cores_physical": rc.host.cpu_cores_physical,
            "cpu_cores_logical": rc.host.cpu_cores_logical,
            "gpu_ready": rc.host.gpu_ready,
            "gpu_name": rc.host.gpu_name,
            "gpu_vram_total_mb": rc.host.gpu_vram_total_mb,
            "gpu_vram_free_mb": rc.host.gpu_vram_free_mb,
        },
        "n_ctx": rc.n_ctx,
        "bonsai_n_batch": rc.bonsai_n_batch,
        "bonsai_n_gpu_layers": rc.bonsai_n_gpu_layers,
        "reranker_max_len": rc.reranker_max_len,
        "reranker_gpu_layers": rc.reranker_gpu_layers,
        "embed_batch": rc.embed_batch,
        "vision_offload": rc.vision_offload,
        "projected_tps": rc.projected_tps,
        "ram_budget_mb": rc.ram_budget_mb,
        "vram_budget_mb": rc.vram_budget_mb,
        "shortfalls": rc.shortfalls,
        "warnings": rc.warnings,
        "suggestions": rc.suggestions,
        "calibration_source": rc.calibration_source,
        "calibration_age_s": rc.calibration_age_s,
    }


def _print_pro_resolved(rc, host) -> None:
    print()
    print("Host")
    print(f"  CPU         {host.cpu_cores_physical} physical / "
          f"{host.cpu_cores_logical} logical cores")
    print(f"  RAM         {host.ram_total_mb} MB total, "
          f"{host.ram_available_mb} MB available")
    print(f"  Disk        {host.disk_free_mb} MB free at cache dir")
    if host.gpu_ready:
        print(f"  GPU         {host.gpu_name or '?'} "
              f"({host.gpu_vram_total_mb} MB total, "
              f"{host.gpu_vram_free_mb} MB free)")
    else:
        print("  GPU         not detected (or supergraph.gpu.setup() not called)")

    print()
    print("Spec")
    for f in rc.spec.__struct_fields__:
        print(f"  {f:<13} {getattr(rc.spec, f)}")

    print()
    print("Resolved")
    print(f"  fits        {'YES' if rc.fits else 'NO'}")
    if rc.fits:
        print(f"  n_ctx       {rc.n_ctx}")
        print(f"  bonsai      n_batch={rc.bonsai_n_batch}, "
              f"n_gpu_layers={rc.bonsai_n_gpu_layers}")
        print(f"  reranker    max_len={rc.reranker_max_len}, "
              f"gpu_layers={rc.reranker_gpu_layers}")
        print(f"  embedder    batch={rc.embed_batch}")
        if rc.projected_tps:
            print("  projected_tps:")
            for cid, tps in rc.projected_tps.items():
                print(f"    {cid:<32} {tps:.1f} tps")
        if rc.ram_budget_mb:
            tot = sum(rc.ram_budget_mb.values())
            print(f"  RAM budget  {tot} MB / {host.ram_available_mb} MB available")
        if rc.vram_budget_mb:
            tot = sum(rc.vram_budget_mb.values())
            print(f"  VRAM budget {tot} MB / {host.gpu_vram_free_mb} MB available")

    if rc.shortfalls:
        print()
        print("Shortfalls")
        for s in rc.shortfalls:
            print(f"  - {s}")
    if rc.suggestions:
        print()
        print("Suggestions")
        for s in rc.suggestions:
            print(f"  - {s}")
    if rc.warnings:
        print()
        print("Warnings")
        for w in rc.warnings:
            print(f"  - {w}")
    print()
    if rc.calibration_source == "missing":
        print("Calibration: missing. Run `supergraph pro setup` (PR#3.5) or "
              "populate ~/.cache/supergraph/calibration.json manually.")
    elif rc.calibration_age_s is not None:
        days = rc.calibration_age_s // 86400
        print(f"Calibration: measured ({days} days old)")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="supergraph", description="supergraph CLI")
    sub = parser.add_subparsers(dest="command")

    pg = sub.add_parser("playground", help="Launch the playground web UI")
    pg.add_argument("--port", type=int, default=7200, help="Port (default 7200)")
    pg.add_argument("--host", default="127.0.0.1", help="Host (default 127.0.0.1)")
    pg.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open browser automatically",
    )
    pg.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to persist playground database",
    )
    pg.set_defaults(func=cmd_playground)

    ie = sub.add_parser("install-embedder", help="Download and install an embedder model")
    ie.add_argument("name", help="Model name (e.g. embeddinggemma-300m)")
    ie.add_argument(
        "--variant",
        default=None,
        help="Model variant (e.g. fp32, q4). Defaults to model's default variant.",
    )
    ie.set_defaults(func=cmd_install_embedder)

    le = sub.add_parser("list-embedders", help="List available and installed embedder models")
    le.set_defaults(func=cmd_list_embedders)

    ue = sub.add_parser("uninstall-embedder", help="Remove an installed embedder model")
    ue.add_argument("name", help="Model name to uninstall")
    ue.set_defaults(func=cmd_uninstall_embedder)

    vis = sub.add_parser("vision", help="Manage the local vision sidecar (serve/stop/status/logs)")
    vis_sub = vis.add_subparsers(dest="vision_command", required=True)
    vis_serve = vis_sub.add_parser("serve", help="Start the sidecar (downloads weights on first run)")
    vis_serve.add_argument("--host", default=None, help="Bind host (env SUPERGRAPH_VISION_HOST, default 127.0.0.1)")
    vis_serve.add_argument("--port", type=int, default=None, help="Bind port (env SUPERGRAPH_VISION_PORT, default 8418)")
    vis_serve.add_argument("--threads", type=int, default=8, help="CPU threads for inference (default 8)")
    vis_serve.add_argument("--n-ctx", type=int, default=4096, help="Context window tokens (default 4096)")
    vis_serve.add_argument("--model", default=None, help="Preset name (see `vision models`); env SUPERGRAPH_VISION_MODEL. Ignored when --repo is given")
    vis_serve.add_argument("--repo", default=None, help="Override: HF repo id (e.g. ggml-org/SmolVLM2-2.2B-Instruct-GGUF)")
    vis_serve.add_argument("--model-file", default=None, help="Override: GGUF filename within repo")
    vis_serve.add_argument("--mmproj-file", default=None, help="Override: mmproj GGUF filename within repo")
    vis_serve.add_argument("--chat-format", default=None, help="Override: llama.cpp chat_format (e.g. llava-1-5, qwen)")
    vis_serve.add_argument("--pull-only", action="store_true", help="Download weights without starting the server")
    vis_sub.add_parser("stop", help="Stop the running sidecar")
    vis_status = vis_sub.add_parser("status", help="Show sidecar status")
    vis_status.add_argument("--host", default=None)
    vis_sub.add_parser("logs", help="Print the sidecar log file to stdout")
    vis_sub.add_parser("models", help="List built-in VLM presets")
    vis.set_defaults(func=cmd_vision)

    pro = sub.add_parser(
        "pro",
        help="Pro mode: spec + host fit check, calibration, status (PR#3+)",
    )
    pro_sub = pro.add_subparsers(dest="pro_command", required=True)
    for name, helptext in (
        ("check",  "Verify spec fits the host using the calibration cache"),
        ("setup",  "Download required models + run live probes"),
        ("probe",  "Re-run live calibration (use after model updates)"),
        ("status", "Show current host + cache state + last probe times"),
    ):
        sp = pro_sub.add_parser(name, help=helptext)
        sp.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON instead of pretty text")
        sp.add_argument("--cache-dir", default=None,
                        help="Calibration cache directory "
                             "(default: ~/.cache/supergraph)")
        sp.add_argument("--embedder", default=None,
                        choices=["jina-v5-small", "jina-v5-nano",
                                 "model2vec-256d", "embeddinggemma-300m",
                                 "fastembed-bge-small", "none"])
        sp.add_argument("--reranker", default=None,
                        choices=["jina-v3", "none"])
        sp.add_argument("--ingest-mode", dest="ingest_mode", default=None,
                        choices=["bonsai", "deterministic"])
        sp.add_argument("--bonsai-quant", dest="bonsai_quant", default=None,
                        choices=["tq1_0", "tq2_0"])
        sp.add_argument("--bonsai-skill", dest="bonsai_skill", default=None,
                        choices=["lite", "full"])
        sp.add_argument("--vision", default=None,
                        choices=["smolvlm2-2.2b", "qwen-vl-3b", "none"])
        sp.add_argument("--audio", default=None,
                        choices=["whisper-tiny", "whisper-base",
                                 "whisper-small", "none"])
        sp.add_argument("--ner", default=None,
                        choices=["tinybert", "none"])
    pro.set_defaults(func=cmd_pro)

    cfg = sub.add_parser("config", help="Show config defaults, schema, or current values")
    cfg.add_argument("--schema", action="store_true", help="Output JSON Schema for supergraph.json")
    cfg.add_argument("--defaults", action="store_true", help="Output all default values as JSON")
    cfg.add_argument("--path", type=str, default=None, help="Path to supergraph.json to show resolved config")
    cfg.set_defaults(func=cmd_config)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
