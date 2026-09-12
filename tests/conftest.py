
from __future__ import annotations

import os as _os

_THREAD_CAP = _os.environ.get("SUPERGRAPH_TEST_BLAS_THREADS", "1")
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "SCIPY_OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
):
    _os.environ.setdefault(_var, _THREAD_CAP)
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("SUPERGRAPH_NER_THREADS", _THREAD_CAP)
_os.environ.setdefault("SUPERGRAPH_EMBED_THREADS", _THREAD_CAP)
_os.environ.setdefault("SUPERGRAPH_RERANK_THREADS", _THREAD_CAP)

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
    _BLAS_LIMIT_CTX = _threadpool_limits(limits=int(_THREAD_CAP))
except Exception:
    _BLAS_LIMIT_CTX = None


import importlib.util
import os

import pytest


_EXTRA_TO_DEP: dict[str, tuple[str, ...]] = {
    "needs_embedder": ("model2vec",),
    "needs_fastembed": ("fastembed",),
    "needs_ingest": ("markitdown", "pymupdf"),
    "needs_vault": ("yaml",),
    "needs_scheduler": ("croniter",),
    "needs_playground": ("fastapi", "pydantic"),
    "needs_gpu": ("onnxruntime",),
    "needs_tui": ("textual",),
}

_FILES_REQUIRING: dict[str, str] = {
    "test_superclaw_tui.py": "needs_tui",
    "test_vault.py": "needs_vault",
    "test_server.py": "needs_playground",
    "test_server_endpoints.py": "needs_playground",
    "test_server_security.py": "needs_playground",
    "test_ingest.py": "needs_ingest",
}


def _is_installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _extras_available() -> dict[str, bool]:
    return {
        marker: all(_is_installed(mod) for mod in deps)
        for marker, deps in _EXTRA_TO_DEP.items()
    }


collect_ignore = [
    fname
    for fname, marker in _FILES_REQUIRING.items()
    if not all(_is_installed(m) for m in _EXTRA_TO_DEP[marker])
]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Include slow tests (pytest.mark.slow). Default is skip. "
             "Also enabled by SUPERGRAPH_RUN_SLOW=1.",
    )


def pytest_configure(config: pytest.Config) -> None:
    for marker in _EXTRA_TO_DEP:
        config.addinivalue_line(
            "markers",
            f"{marker}: skipped unless the matching supergraph extra is installed",
        )
    config.addinivalue_line(
        "markers",
        "slow: skipped unless --run-slow or SUPERGRAPH_RUN_SLOW=1 is set",
    )


def _slow_enabled(config: pytest.Config) -> bool:
    return (
        bool(config.getoption("--run-slow"))
        or os.environ.get("SUPERGRAPH_RUN_SLOW") == "1"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    available = _extras_available()
    run_slow = _slow_enabled(config)
    for item in items:
        if not run_slow and item.get_closest_marker("slow") is not None:
            item.add_marker(
                pytest.mark.skip(reason="slow; pass --run-slow to include")
            )
        for marker, ok in available.items():
            if ok:
                continue
            if item.get_closest_marker(marker) is None:
                continue
            missing = ", ".join(_EXTRA_TO_DEP[marker])
            item.add_marker(
                pytest.mark.skip(reason=f"requires optional deps: {missing}")
            )


@pytest.fixture(scope="session", autouse=True)
def _blas_cap_session_guard():
    cap = int(_THREAD_CAP)
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=cap):
            yield
    except ImportError:
        yield
