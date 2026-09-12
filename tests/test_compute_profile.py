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
    cp.configure()
    yield
    _apply_session_lock()


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
    patches = _host()
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


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


@pytest.mark.parametrize("load,battery,disable_load,disable_bat,lock_embed,expected", [
    (  5.0, False, False, False, None, 4),
    ( 70.0, False, False, False, None, 2),
    ( 40.0, False, False, False, None, 4),
    ( 70.0, False, True,  False, None, 4),
    ( 70.0, False, False, False, 3,    3),
    (  5.0, True,  False, False, None, 3),
    (  5.0, True,  False, True,  None, 4),
    (  5.0, True,  False, False, 4,    4),
    ( 70.0, True,  False, False, None, 1),
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
    with (
        patch.object(cp, "_detect_load_pct", return_value=95.0),
        patch.object(cp, "_detect_battery", return_value=True),
    ):
        cp.get_profile.cache_clear()
        assert cp.get_profile().ner_threads == 2


@pytest.mark.parametrize("config_val,env_val,expected", [
    (3,    "7",  3),
    (None, "7",  7),
    (None, None, 4),
    (0,    None, 1),
    (-5,   None, 1),
])
def test_embed_threads_precedence(desktop_host, monkeypatch, config_val, env_val, expected):
    if env_val is not None:
        monkeypatch.setenv("SUPERGRAPH_EMBED_THREADS", env_val)
    cp.configure(embed_threads=config_val)
    assert cp.get_profile().embed_threads == expected


@pytest.mark.parametrize("config_profile,env_profile,expected_name", [
    ("tiny",   None,     "tiny"),
    (None,     "laptop", "laptop"),
    ("tiny",   "laptop", "tiny"),
    (None,     None,     "desktop"),
])
def test_profile_tier_precedence(desktop_host, monkeypatch, config_profile, env_profile, expected_name):
    if env_profile:
        monkeypatch.setenv("SUPERGRAPH_PROFILE", env_profile)
    cp.configure(profile=config_profile)
    assert cp.get_profile().name == expected_name


def test_embed_batch_override(desktop_host):
    cp.configure(embed_batch_size=256)
    assert cp.get_profile().embed_batch_size == 256


def test_reconfigure_invalidates_cache(desktop_host):
    cp.configure(embed_threads=2)
    assert cp.get_profile().embed_threads == 2

    cp.configure(embed_threads=6)
    assert cp.get_profile().embed_threads == 6

    cp.configure()
    assert cp.get_profile().embed_threads == 4


@pytest.mark.parametrize("gpu_env,detect_return,expected_has_gpu,expected_name", [
    (None, (False, None),                      False, "desktop"),
    ("1",  (True, "CUDAExecutionProvider"),    True,  "gpu"),
    (None, (True, "CUDAExecutionProvider"),    False, "desktop"),
])
def test_gpu_detection_requires_opt_in(desktop_host, monkeypatch, gpu_env, detect_return, expected_has_gpu, expected_name):
    if gpu_env:
        monkeypatch.setenv("SUPERGRAPH_GPU", gpu_env)
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
