#!/usr/bin/env python3
"""Environment manifest for autoresearch prompts.

Queries the live Python environment for every package on the algos
allowlist and reports exact installed versions. This is the context
autoresearch feeds the LLM so rewrites don't hallucinate packages or
pick wrong APIs.

Contract:
    - stdout = environment manifest in the chosen format (--markdown or --json)
    - lists only packages that are actually importable AND pinned
    - CORE packages are always required; missing → non-zero exit
    - OPTIONAL packages report version if installed, 'not installed' otherwise

Usage:
    python -m benchmarks.algos.dump_env                # markdown (default)
    python -m benchmarks.algos.dump_env --json         # json output
    python -m benchmarks.algos.dump_env --compact      # one line per pkg
    python -m benchmarks.algos.dump_env --allowlist    # print raw allowlist

Autoresearch wiring:
    - Run once per session to produce ENVIRONMENT.md
    - Include the markdown as system/context prompt to the LLM
    - LLM knows exact names + versions it can import
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from benchmarks.algos.allowlist import (
    CORE,
    FORBIDDEN_PREFIXES,
    OPTIONAL,
    STDLIB,
)


def _installed_version(pypi_name: str) -> str | None:
    try:
        return importlib.metadata.version(pypi_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _importable(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def build_manifest() -> dict:
    py = sys.version_info
    manifest = {
        "python": f"{py.major}.{py.minor}.{py.micro}",
        "platform": platform.platform(),
        "core": {},
        "optional_installed": {},
        "optional_missing": {},
        "stdlib_allowed": sorted(STDLIB),
        "forbidden_prefixes": sorted(FORBIDDEN_PREFIXES),
    }

    for mod, meta in CORE.items():
        ver = _installed_version(meta["pypi"])
        importable = _importable(mod)
        manifest["core"][mod] = {
            "pypi": meta["pypi"],
            "version": ver,
            "import_name": mod,
            "importable": importable,
            "hint": meta["hint"],
        }

    for mod, meta in OPTIONAL.items():
        ver = _installed_version(meta["pypi"])
        importable = _importable(mod)
        entry = {
            "pypi": meta["pypi"],
            "version": ver,
            "import_name": mod,
            "importable": importable,
            "hint": meta["hint"],
        }
        if importable and ver is not None:
            manifest["optional_installed"][mod] = entry
        else:
            manifest["optional_missing"][mod] = entry

    return manifest


def render_markdown(manifest: dict) -> str:
    lines = []
    lines.append("# supergraph/algos - environment manifest")
    lines.append("")
    lines.append(f"- python: **{manifest['python']}**")
    lines.append(f"- platform: {manifest['platform']}")
    lines.append("")
    lines.append("> This file lists the **exact packages and versions**")
    lines.append("> available when rewriting a file in `supergraph/algos/`.")
    lines.append("> Do not import anything not listed here - the purity")
    lines.append("> gate will reject the patch.")
    lines.append("")

    lines.append("## Core (always available, always allowed)")
    lines.append("")
    for mod, entry in manifest["core"].items():
        ver = entry["version"] or "MISSING"
        lines.append(f"- **`{entry['pypi']}=={ver}`** - `import {entry['import_name']}`")
        lines.append(f"  - {entry['hint']}")
    lines.append("")

    lines.append("## Optional (installed - you may use any of these)")
    lines.append("")
    if manifest["optional_installed"]:
        for mod, entry in manifest["optional_installed"].items():
            lines.append(f"- **`{entry['pypi']}=={entry['version']}`** - `import {entry['import_name']}`")
            lines.append(f"  - {entry['hint']}")
    else:
        lines.append("_(none installed)_")
    lines.append("")

    if manifest["optional_missing"]:
        lines.append("## Optional (NOT installed - do NOT import)")
        lines.append("")
        for mod, entry in manifest["optional_missing"].items():
            lines.append(f"- `{entry['pypi']}` - not available in this environment")
        lines.append("")

    lines.append("## Python standard library (allowed)")
    lines.append("")
    lines.append(", ".join(f"`{m}`" for m in manifest["stdlib_allowed"]))
    lines.append("")

    lines.append("## Forbidden imports (purity gate rejects these)")
    lines.append("")
    lines.append(", ".join(f"`{p}`" for p in manifest["forbidden_prefixes"]))
    lines.append("")

    lines.append("## Rules")
    lines.append("")
    lines.append("1. Import only from the lists above")
    lines.append("2. No I/O (files, sockets, SQLite, network)")
    lines.append("3. No logging or global state")
    lines.append("4. Pure functions - do not mutate caller state unless documented")
    lines.append("5. Deterministic - do not read wall-clock time inside algos")
    lines.append("6. Type-hint the public API")
    lines.append("7. Preserve function signatures so existing callers keep working")
    lines.append("")
    return "\n".join(lines)


def render_compact(manifest: dict) -> str:
    lines = []
    lines.append(f"# python {manifest['python']}")
    lines.append("# core (required):")
    for mod, entry in manifest["core"].items():
        ver = entry["version"] or "MISSING"
        lines.append(f"{entry['pypi']}=={ver}  # import {mod}")
    lines.append("# optional (installed):")
    for mod, entry in manifest["optional_installed"].items():
        lines.append(f"{entry['pypi']}=={entry['version']}  # import {mod}")
    lines.append("# forbidden:")
    lines.append(", ".join(sorted(manifest["forbidden_prefixes"])))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--json",
        action="store_true",
        help="emit machine-parseable JSON",
    )
    group.add_argument(
        "--compact",
        action="store_true",
        help="emit one-line-per-package requirement-style output",
    )
    group.add_argument(
        "--allowlist",
        action="store_true",
        help="print the raw allowlist (useful for checking additions)",
    )
    parser.add_argument(
        "--write",
        type=str,
        default=None,
        help="write output to this file instead of stdout",
    )
    args = parser.parse_args()

    if args.allowlist:
        payload = {
            "core": CORE,
            "optional": OPTIONAL,
            "stdlib": sorted(STDLIB),
            "forbidden_prefixes": sorted(FORBIDDEN_PREFIXES),
        }
        out = json.dumps(payload, indent=2)
    else:
        manifest = build_manifest()
        missing_core = [
            mod for mod, e in manifest["core"].items()
            if e["version"] is None or not e["importable"]
        ]
        if missing_core:
            print(
                f"ERROR: required core packages missing: {missing_core}",
                file=sys.stderr,
            )
            return 1

        if args.json:
            out = json.dumps(manifest, indent=2)
        elif args.compact:
            out = render_compact(manifest)
        else:
            out = render_markdown(manifest)

    if args.write:
        Path(args.write).write_text(out + ("\n" if not out.endswith("\n") else ""))
        print(f"wrote {args.write}", file=sys.stderr)
    else:
        sys.stdout.write(out)
        if not out.endswith("\n"):
            sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
