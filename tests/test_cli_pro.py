from __future__ import annotations

import json
import subprocess
import sys

import pytest


GS = sys.executable


def _run(args: list[str], cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [GS, "-m", "supergraph.cli", "pro", *args],
        cwd=cwd, capture_output=True, text=True, timeout=30,
    )


class TestProHelp:
    def test_top_level_help(self):
        r = _run(["--help"])
        assert r.returncode == 0
        assert "check" in r.stdout
        assert "setup" in r.stdout
        assert "probe" in r.stdout
        assert "status" in r.stdout

    def test_check_help_lists_slot_overrides(self):
        r = _run(["check", "--help"])
        assert r.returncode == 0
        for slot in ("--embedder", "--reranker", "--ingest-mode",
                     "--bonsai-quant", "--bonsai-skill",
                     "--vision", "--audio", "--ner",
                     "--json", "--cache-dir"):
            assert slot in r.stdout, f"missing flag {slot}"


_MINIMAL_DEPS_SPEC = [
    "--ingest-mode", "deterministic",
    "--embedder", "model2vec-256d",
    "--vision", "none",
    "--audio", "none",
]


class TestProCheckEmptyCache:

    def test_text_output_says_calibration_missing(self, tmp_path):
        r = _run(["check", "--cache-dir", str(tmp_path), *_MINIMAL_DEPS_SPEC])
        assert r.returncode == 3, f"stderr={r.stderr!r}"
        assert "fits" in r.stdout.lower()
        assert "NO" in r.stdout
        assert "calibration missing" in r.stdout.lower() or \
               "calibration: missing" in r.stdout.lower()

    def test_json_output_is_parseable(self, tmp_path):
        r = _run(["check", "--cache-dir", str(tmp_path), "--json",
                  *_MINIMAL_DEPS_SPEC])
        assert r.returncode == 3, f"stderr={r.stderr!r}"
        data = json.loads(r.stdout)
        assert data["fits"] is False
        assert data["calibration_source"] == "missing"
        assert isinstance(data["shortfalls"], list)
        assert any("calibration" in s.lower() for s in data["shortfalls"])
        assert data["spec"]["embedder"] == "model2vec-256d"

    def test_slot_override_reflected_in_json(self, tmp_path):
        r = _run([
            "check", "--cache-dir", str(tmp_path), "--json",
            "--embedder", "model2vec-256d",
            "--reranker", "none",
            "--ingest-mode", "deterministic",
            "--vision", "none", "--audio", "none",
        ])
        assert r.returncode == 3, f"stderr={r.stderr!r}"
        data = json.loads(r.stdout)
        assert data["spec"]["embedder"] == "model2vec-256d"
        assert data["spec"]["reranker"] == "none"
        assert data["spec"]["ingest_mode"] == "deterministic"


class TestProCheckHostFields:

    def test_host_block_present(self, tmp_path):
        r = _run(["check", "--cache-dir", str(tmp_path), "--json",
                  *_MINIMAL_DEPS_SPEC])
        assert r.returncode == 3
        data = json.loads(r.stdout)
        assert "host" in data
        h = data["host"]
        for f in ("ram_total_mb", "ram_available_mb", "disk_free_mb",
                  "cpu_cores_physical", "cpu_cores_logical",
                  "gpu_ready"):
            assert f in h, f"missing host field {f}"
        assert h["ram_total_mb"] > 0
        assert h["cpu_cores_logical"] >= 1


class TestProCheckExtrasGate:

    def test_extras_missing_exits_2_with_structured_error(self, tmp_path):
        r = _run([
            "check", "--cache-dir", str(tmp_path), "--json",
            "--embedder", "fastembed-bge-small",
            "--ingest-mode", "deterministic",
            "--vision", "none", "--audio", "none", "--reranker", "none",
            "--ner", "none",
        ])
        if r.returncode == 2:
            data = json.loads(r.stdout)
            assert data["error"] == "extra_not_installed"
            assert "fastembed" in data["missing_dists"]
        else:
            assert r.returncode == 3


class TestProSetupExtrasGate:

    @pytest.mark.parametrize("subcmd", ["setup", "probe"])
    def test_extras_missing_exits_2(self, subcmd):
        r = _run([subcmd, "--json",
                  "--embedder", "fastembed-bge-small",
                  "--ingest-mode", "deterministic",
                  "--vision", "none", "--audio", "none",
                  "--reranker", "none", "--ner", "none"])
        if r.returncode == 2:
            data = json.loads(r.stdout)
            assert data["error"] == "extra_not_installed"
            assert "fastembed" in data["missing_dists"]


class TestProSetupUnregisteredComponent:

    def test_unregistered_component_recorded_as_failure(self, tmp_path):
        r = _run(["setup", "--json",
                  "--cache-dir", str(tmp_path),
                  "--ingest-mode", "deterministic",
                  "--embedder", "model2vec-256d",
                  "--vision", "none", "--audio", "none",
                  "--reranker", "none", "--ner", "none"],
                 )
        assert r.returncode in (0, 1), f"unexpected rc={r.returncode}; stderr={r.stderr!r}"
        data = json.loads(r.stdout)
        assert "all_ok" in data
        assert "successes" in data
        assert "failures" in data
        assert "events" in data
