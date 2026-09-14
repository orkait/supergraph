from __future__ import annotations

import shutil

from superclaw.settings import HOST_TOOLS

which = shutil.which


def host_tools() -> tuple[str, ...]:
    return tuple(name for name, _ in HOST_TOOLS if which(name))


def guidance(found: tuple[str, ...]) -> str:
    if not found:
        return ""
    swaps = [f"{name} over {legacy}" for name, legacy in HOST_TOOLS if legacy and name in found]
    line = f"Host tools present: {', '.join(found)}."
    return f"{line} In bash prefer {', '.join(swaps)}." if swaps else line
