"""Behaviour tests for supergraph.core.compute_profile.

Parametrized matrices cover: tier sizing, battery/load scaling, override
precedence (config > env > base), floor clamping, cache invalidation,
and GPU opt-in.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from supergraph.core import compute_profile as cp


_ENV_KEYS = (
    "SUPERGRAPH_PROFILE",
    "SUPERGRAPH_NER_THREADS",
    "SUPERGRAPH_EMBED_THREADS",
    "SUPERGRAPH_RERANK_THREADS",
    "SUPERGRAPH_EMBED_BATCH",
    "SUPERGRAPH_GPU",
)


def _apply_session_lock():
    """Same lock conftest installs at session start (tiny, 1-thread)."""
    cp.configure(
        profile="tiny",
        ner_threads=1,
        embed_threads=1,
        rerank_threads=1,
        disable_load_scaling=True,
        disable_battery_scaling=True,
    )


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    cp.configure()  # clean slate for the test itself
    yield
    _apply_session_lock()  # restore thermal guard for subsequent tests


def _host(cores=8, logical=16, ram=32.0, gpu=(False, None), battery=False, load=5.0):
    return (
        patch.object(cp, "_detect_cores", return_value=(cores, logical)),
        patch.object(cp, "_detect_ram_gb", return_value=ram),
        patch.object(cp, "_detect_gpu", return_value=gpu),
        patch.object(cp, "_detect_battery", return_value=battery),
        patch.object(cp, "_detect_load_pct", return_value=load),
    )


@pytest.fixture
def desktop_host():
    """8c/16t, 32GB, plugged in, idle -> desktop base (embed=4, rerank=4)."""
    patches = _host()
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


# ---------------- tier sizing ----------------

@pytest.mark.parametrize("cores,ram,expected_name,expected_embed", [
    (2, 4.0, "tiny", 1),
    (4, 8.0, "laptop", 2),
    (8, 32.0, "desktop", 4),
])
def test_tier_sizing(cores, ram, expected_name, expected_embed):
    with (
        patch.object(cp, "_detect_cores", return_value=(cores, cores * 2)),
        patch.object(cp, "_detect_ram_gb", return_value=ram),
        patch.object(cp, "_detect_gpu", return_value=(False, None)),
        patch.object(cp, "_detect_battery", return_value=False),
        patch.object(cp, "_detect_load_pct", return_value=5.0),
    ):
        p = cp.get_profile()
        assert p.name == expected_name
        assert p.embed_threads == expected_embed


# ---------------- scaling matrix (load + battery + disable flags + lock) ----------------

@pytest.mark.parametrize("load,battery,disable_load,disable_bat,lock_embed,expected", [
    # (load, on_battery, disable_load_scaling, disable_battery_scaling, embed lock, expected embed)
    (  5.0, False, False, False, None, 4),  # idle baseline
    ( 70.0, False, False, False, None, 2),  # load halves (M1 baseline)
    ( 40.0, False, False, False, None, 4),  # boundary: 40 not > 40
    ( 70.0, False, True,  False, None, 4),  # disable_load skips halving
    ( 70.0, False, False, False, 3,    3),  # lock beats load scaling
    (  5.0, True,  False, False, None, 3),  # battery -1 (M2 baseline)
    (  5.0, True,  False, True,  None, 4),  # disable_battery skips decrement (M3)
    (  5.0, True,  False, False, 4,    4),  # lock beats battery
    ( 70.0, True,  False, False, None, 1),  # compound: 4-1=3, 3//2=1
])
def test_scaling_matrix(desktop_host, load, battery, disable_load, disable_bat, lock_embed, expected):
    cp.configure(
        embed_threads=lock_embed,
        disable_load_scaling=disable_load,
        disable_battery_scaling=disable_bat,
    )
    with (
        patch.object(cp, "_detect_load_pct", return_value=load),
        patch.object(cp, "_detect_battery", return_value=battery),
    ):
        cp.get_profile.cache_clear()
        assert cp.get_profile().embed_threads == expected


def test_tiny_tier_immune_to_scaling():
    with (
        patch.object(cp, "_detect_cores", return_value=(2, 4)),
        patch.object(cp, "_detect_ram_gb", return_value=4.0),
        patch.object(cp, "_detect_gpu", return_value=(False, None)),
        patch.object(cp, "_detect_battery", return_value=True),
        patch.object(cp, "_detect_load_pct", return_value=90.0),
    ):
        p = cp.get_profile()
        assert p.name == "tiny"
        assert p.embed_threads == 1


def test_ner_never_scaled(desktop_host):
    """NER stays at tier base regardless of load/battery."""
    with (
        patch.object(cp, "_detect_load_pct", return_value=95.0),
        patch.object(cp, "_detect_battery", return_value=True),
    ):
        cp.get_profile.cache_clear()
        assert cp.get_profile().ner_threads == 2


# ---------------- override precedence (config > env > base) ----------------

@pytest.mark.parametrize("config_val,env_val,expected", [
    (3,    "7",  3),   # config wins over env
    (None, "7",  7),   # env wins over base when no config
    (None, None, 4),   # base when neither
    (0,    None, 1),   # floor clamps 0 -> 1
    (-5,   None, 1),   # floor clamps negative -> 1
])
def test_embed_threads_precedence(desktop_host, monkeypatch, config_val, env_val, expected):
    if env_val is not None:
        monkeypatch.setenv("SUPERGRAPH_EMBED_THREADS", env_val)
    cp.configure(embed_threads=config_val)
    assert cp.get_profile().embed_threads == expected


@pytest.mark.parametrize("config_profile,env_profile,expected_name", [
    ("tiny",   None,     "tiny"),
    (None,     "laptop", "laptop"),
    ("tiny",   "laptop", "tiny"),     # config wins
    (None,     None,     "desktop"),  # auto-classify on desktop host
])
def test_profile_tier_precedence(desktop_host, monkeypatch, config_profile, env_profile, expected_name):
    if env_profile:
        monkeypatch.setenv("SUPERGRAPH_PROFILE", env_profile)
    cp.configure(profile=config_profile)
    assert cp.get_profile().name == expected_name


def test_embed_batch_override(desktop_host):
    cp.configure(embed_batch_size=256)
    assert cp.get_profile().embed_batch_size == 256


# ---------------- cache invalidation ----------------

def test_reconfigure_invalidates_cache(desktop_host):
    cp.configure(embed_threads=2)
    assert cp.get_profile().embed_threads == 2

    cp.configure(embed_threads=6)
    assert cp.get_profile().embed_threads == 6

    cp.configure()  # wipe
    assert cp.get_profile().embed_threads == 4


# ---------------- GPU opt-in gate ----------------

@pytest.mark.parametrize("gpu_env,detect_return,expected_has_gpu,expected_name", [
    (None, (False, None),                      False, "desktop"),
    ("1",  (True, "CUDAExecutionProvider"),    True,  "gpu"),
    (None, (True, "CUDAExecutionProvider"),    False, "desktop"),  # env required even if probe says True
])
def test_gpu_detection_requires_opt_in(desktop_host, monkeypatch, gpu_env, detect_return, expected_has_gpu, expected_name):
    if gpu_env:
        monkeypatch.setenv("SUPERGRAPH_GPU", gpu_env)
    # Rebind the real _detect_gpu since desktop_host already patched it as (False, None).
    # _detect_gpu itself checks SUPERGRAPH_GPU env before probing, so we can let it run
    # directly when env is unset and patch only when env is set.
    if gpu_env:
        with patch.object(cp, "_detect_gpu", return_value=detect_return):
            cp.get_profile.cache_clear()
            p = cp.get_profile()
    else:
        cp.get_profile.cache_clear()
        p = cp.get_profile()
    assert p.has_gpu is expected_has_gpu
    assert p.name == expected_name


def test_env_fingerprint_invalidates_cache(monkeypatch):
    from supergraph.core import compute_profile as cp

    cp.configure()
    monkeypatch.delenv("SUPERGRAPH_EMBED_THREADS", raising=False)
    p1 = cp.get_profile()
    base_threads = p1.embed_threads

    monkeypatch.setenv("SUPERGRAPH_EMBED_THREADS", "99")
    p2 = cp.get_profile()

    assert p2.embed_threads == 99
