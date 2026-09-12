
import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ALGOS_DIR = REPO_ROOT / "src" / "supergraph" / "algos"
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks.algos.allowlist import (  # noqa: E402
    FORBIDDEN_PREFIXES,
    allowed_import_names,
    is_forbidden,
)


def _collect_imports(source: str) -> list[str]:
    tree = ast.parse(source)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
    return modules


def _algo_files() -> list[Path]:
    if not ALGOS_DIR.exists():
        return []
    return [p for p in ALGOS_DIR.glob("*.py") if p.name != "__init__.py"]


@pytest.mark.parametrize("path", _algo_files(), ids=lambda p: p.name)
def test_algos_file_has_only_pure_imports(path: Path):
    source = path.read_text()
    modules = _collect_imports(source)
    allowed = allowed_import_names()
    for m in modules:
        top = m.split(".")[0]
        assert not is_forbidden(m), (
            f"{path.name}: forbidden import {m!r} "
            f"(prefix {top!r} is on FORBIDDEN_PREFIXES in benchmarks/algos/allowlist.py)"
        )
        assert top in allowed, (
            f"{path.name}: unexpected import {m!r} - "
            f"add {top!r} to benchmarks/algos/allowlist.py "
            f"(STDLIB, CORE, or OPTIONAL) if it's genuinely pure"
        )


def test_algos_init_is_pure():
    init = ALGOS_DIR / "__init__.py"
    if not init.exists():
        pytest.skip("algos package not present")
    source = init.read_text()
    modules = _collect_imports(source)
    for m in modules:
        assert not is_forbidden(m), f"__init__.py: forbidden import {m!r}"


def test_algos_dir_exists():
    assert ALGOS_DIR.exists(), f"algos directory missing at {ALGOS_DIR}"


def test_allowlist_is_importable():
    assert callable(allowed_import_names)
    names = allowed_import_names()
    assert "numpy" in names
    assert "scipy" in names
    assert "math" in names
    assert FORBIDDEN_PREFIXES
    assert "supergraph" in FORBIDDEN_PREFIXES
