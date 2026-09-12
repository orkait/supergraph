from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from supergraph.pro import (
    CalibrationCache, CalibrationEntry, HostSnapshot, _DEFAULT_CACHE_DIR,
    _EXTRA_EMBED_BATCH, _EXTRA_N_CTX_DEFAULT, _EXTRA_N_CTX_MAX,
    _EXTRA_N_CTX_MIN, _EXTRA_N_BATCH, _EXTRA_RERANKER_MAX,
)

_log = logging.getLogger(__name__)

_PROBE_TEXT = (
    "supergraph pro calibration probe. The system is recording resident "
    "memory and tokens-per-second on this host. This sentence is "
    "deliberately ordinary so the embedder, reranker, and ingester all "
    "see workload representative of typical agent-conversation traffic."
)
_PROBE_TEXTS_BATCH = [_PROBE_TEXT] * 100
_PROBE_QUERY_DOC_PAIRS = [
    (_PROBE_TEXT, _PROBE_TEXT[i:] + " " + _PROBE_TEXT[:i])
    for i in range(0, 50, 1)
]


@dataclass
class ProbeResult:

    component_id: str
    duration_s: float
    entry: CalibrationEntry | None = None
    error: str | None = None


def _process_rss_mb() -> int:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        return 0


def _vram_free_mb() -> int:
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode != 0:
            return 0
        first = out.stdout.strip().splitlines()[0].strip()
        return int(first)
    except Exception:
        return 0


def _free_caches() -> None:
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _measure_callable_tps(
    callable_: Callable[[], int],
    n_iters: int = 3,
) -> float:
    timings: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        tokens = int(callable_())
        dt = time.perf_counter() - t0
        if dt > 0 and tokens > 0:
            timings.append(tokens / dt)
    if not timings:
        return 0.0
    timings.sort()
    return timings[len(timings) // 2]


class Probe:

    component_id: str = ""

    def download(self, cache_dir: Path, host: HostSnapshot) -> int:
        raise NotImplementedError

    def measure(
        self,
        cache_dir: Path,
        host: HostSnapshot,
        disk_mb: int,
    ) -> CalibrationEntry:
        raise NotImplementedError

    def run(
        self,
        cache_dir: Path,
        host: HostSnapshot,
    ) -> ProbeResult:
        t0 = time.perf_counter()
        try:
            disk_mb = self.download(cache_dir, host)
        except Exception as e:
            return ProbeResult(
                component_id=self.component_id,
                duration_s=time.perf_counter() - t0,
                error=f"download failed: {type(e).__name__}: {e}",
            )
        try:
            entry = self.measure(cache_dir, host, disk_mb)
        except Exception as e:
            return ProbeResult(
                component_id=self.component_id,
                duration_s=time.perf_counter() - t0,
                error=f"measure failed: {type(e).__name__}: {e}",
            )
        finally:
            _free_caches()
        return ProbeResult(
            component_id=self.component_id,
            duration_s=time.perf_counter() - t0,
            entry=entry,
        )


class Model2VecProbe(Probe):

    component_id = "embedder:model2vec-256d"
    model_name = "minishlab/M2V_base_output"

    def download(self, cache_dir: Path, host: HostSnapshot) -> int:
        return 0

    def measure(
        self,
        cache_dir: Path,
        host: HostSnapshot,
        disk_mb: int,
    ) -> CalibrationEntry:
        from supergraph.embedding.model2vec_embedder import Model2VecEmbedder

        rss_baseline = _process_rss_mb()
        embedder = Model2VecEmbedder(model_name=self.model_name)

        rss_after_load = _process_rss_mb()
        ram_idle = max(rss_after_load - rss_baseline, 0)

        peak_rss = rss_after_load

        def _embed_batch():
            nonlocal peak_rss
            vecs = embedder.encode_documents(_PROBE_TEXTS_BATCH)
            peak_rss = max(peak_rss, _process_rss_mb())
            return len(vecs) * 8

        tps = _measure_callable_tps(_embed_batch, n_iters=3)
        ram_at_default = max(peak_rss - rss_baseline, 0)

        try:
            from huggingface_hub import scan_cache_dir
            cache = scan_cache_dir()
            for repo in cache.repos:
                if self.model_name.split("/")[-1] in str(repo.repo_id):
                    disk_mb = int(repo.size_on_disk / (1024 * 1024))
                    break
        except Exception:
            pass

        return CalibrationEntry(
            component_id=self.component_id,
            measured_at=datetime.now(timezone.utc).isoformat(),
            ram_mb_idle=ram_idle,
            ram_mb_at_default=ram_at_default,
            ram_mb_min=ram_idle,
            ram_mb_max=ram_at_default,
            disk_mb=disk_mb,
            vram_mb_full_offload=0,
            tps_cpu_threads={str(host.cpu_cores_logical): tps},
            tps_gpu_full_offload=None,
            extra={_EXTRA_EMBED_BATCH: 64},
        )


class TinyBERTNERProbe(Probe):

    component_id = "ner:tinybert"

    def download(self, cache_dir: Path, host: HostSnapshot) -> int:
        from supergraph.config import SuperGraphConfig
        cfg = SuperGraphConfig()
        model_dir = Path(cfg.dsl.entity_model_dir or "./models/tinybert-ner")
        if not model_dir.is_dir():
            raise RuntimeError(
                f"tinybert model not found at {model_dir}. Install via "
                "`supergraph install-embedder tinybert` or set "
                "config.dsl.entity_model_dir to the directory containing "
                "model.onnx + tokenizer.json."
            )
        total = sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file())
        return int(total / (1024 * 1024))

    def measure(
        self,
        cache_dir: Path,
        host: HostSnapshot,
        disk_mb: int,
    ) -> CalibrationEntry:
        from supergraph.config import SuperGraphConfig
        from supergraph.ingest.entity_extract import extract_batch
        cfg = SuperGraphConfig()
        model_dir = cfg.dsl.entity_model_dir or "./models/tinybert-ner"

        rss_baseline = _process_rss_mb()
        _ = extract_batch([_PROBE_TEXT], model_dir=model_dir,
                         max_length=256, score_threshold=0.6)
        rss_after_load = _process_rss_mb()
        ram_idle = max(rss_after_load - rss_baseline, 0)

        peak_rss = rss_after_load
        sentences = [_PROBE_TEXT] * 100

        def _ner_batch():
            nonlocal peak_rss
            extract_batch(sentences, model_dir=model_dir,
                         max_length=256, score_threshold=0.6)
            peak_rss = max(peak_rss, _process_rss_mb())
            return len(sentences) * 32

        tps = _measure_callable_tps(_ner_batch, n_iters=3)
        ram_at_default = max(peak_rss - rss_baseline, 0)

        return CalibrationEntry(
            component_id=self.component_id,
            measured_at=datetime.now(timezone.utc).isoformat(),
            ram_mb_idle=ram_idle,
            ram_mb_at_default=ram_at_default,
            ram_mb_min=ram_idle,
            ram_mb_max=ram_at_default,
            disk_mb=disk_mb,
            vram_mb_full_offload=0,
            tps_cpu_threads={str(host.cpu_cores_logical): tps},
            tps_gpu_full_offload=None,
            extra={},
        )


class JinaOnnxEmbedderProbe(Probe):

    def __init__(self, slot_id: str, install_name: str, dims: int):
        self.component_id = f"embedder:{slot_id}"
        self._install_name = install_name
        self._dims = dims

    def download(self, cache_dir: Path, host: HostSnapshot) -> int:
        from supergraph.registry.installer import install_embedder
        target = install_embedder(self._install_name)
        total = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
        return int(total / (1024 * 1024))

    def measure(
        self,
        cache_dir: Path,
        host: HostSnapshot,
        disk_mb: int,
    ) -> CalibrationEntry:
        from supergraph.registry.installer import load_installed_embedder

        rss_baseline = _process_rss_mb()
        embedder = load_installed_embedder(self._install_name, dims=self._dims)
        rss_after_load = _process_rss_mb()
        ram_idle = max(rss_after_load - rss_baseline, 0)

        peak_rss = rss_after_load

        def _embed_batch():
            nonlocal peak_rss
            vecs = embedder.encode_documents(_PROBE_TEXTS_BATCH)
            peak_rss = max(peak_rss, _process_rss_mb())
            return len(vecs) * 8

        tps = _measure_callable_tps(_embed_batch, n_iters=3)
        ram_at_default = max(peak_rss - rss_baseline, 0)

        return CalibrationEntry(
            component_id=self.component_id,
            measured_at=datetime.now(timezone.utc).isoformat(),
            ram_mb_idle=ram_idle,
            ram_mb_at_default=ram_at_default,
            ram_mb_min=ram_idle,
            ram_mb_max=ram_at_default,
            disk_mb=disk_mb,
            vram_mb_full_offload=0,
            tps_cpu_threads={str(host.cpu_cores_logical): tps},
            tps_gpu_full_offload=None,
            extra={_EXTRA_EMBED_BATCH: 64},
        )


class FastembedProbe(Probe):

    component_id = "embedder:fastembed-bge-small"

    def download(self, cache_dir: Path, host: HostSnapshot) -> int:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise RuntimeError(
                "fastembed not installed; pip install 'supergraph[embedders-extra]'"
            ) from e
        TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        try:
            from huggingface_hub import scan_cache_dir
            cache = scan_cache_dir()
            for repo in cache.repos:
                if "bge-small" in str(repo.repo_id):
                    return int(repo.size_on_disk / (1024 * 1024))
        except Exception:
            pass
        return 0

    def measure(
        self,
        cache_dir: Path,
        host: HostSnapshot,
        disk_mb: int,
    ) -> CalibrationEntry:
        from fastembed import TextEmbedding

        rss_baseline = _process_rss_mb()
        emb = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        rss_after_load = _process_rss_mb()
        ram_idle = max(rss_after_load - rss_baseline, 0)

        peak_rss = rss_after_load

        def _embed_batch():
            nonlocal peak_rss
            list(emb.embed(_PROBE_TEXTS_BATCH))
            peak_rss = max(peak_rss, _process_rss_mb())
            return len(_PROBE_TEXTS_BATCH) * 8

        tps = _measure_callable_tps(_embed_batch, n_iters=3)
        ram_at_default = max(peak_rss - rss_baseline, 0)

        return CalibrationEntry(
            component_id=self.component_id,
            measured_at=datetime.now(timezone.utc).isoformat(),
            ram_mb_idle=ram_idle,
            ram_mb_at_default=ram_at_default,
            ram_mb_min=ram_idle,
            ram_mb_max=ram_at_default,
            disk_mb=disk_mb,
            vram_mb_full_offload=0,
            tps_cpu_threads={str(host.cpu_cores_logical): tps},
            tps_gpu_full_offload=None,
            extra={_EXTRA_EMBED_BATCH: 64},
        )


class JinaV3RerankerProbe(Probe):

    component_id = "reranker:jina-v3"

    def download(self, cache_dir: Path, host: HostSnapshot) -> int:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface-hub not installed; pip install 'supergraph[pro]'"
            ) from e
        from supergraph.config import SuperGraphConfig
        cfg = SuperGraphConfig()
        model_path = Path(cfg.dsl.reranker_model_dir or "")
        proj_path = Path(cfg.dsl.reranker_projector_path or "")
        target_dir = model_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        repo = "jinaai/jina-reranker-v3-GGUF"
        gguf = hf_hub_download(
            repo_id=repo, filename=model_path.name, local_dir=str(target_dir),
        )
        proj = hf_hub_download(
            repo_id=repo, filename=proj_path.name, local_dir=str(target_dir),
        )
        return int((Path(gguf).stat().st_size + Path(proj).stat().st_size)
                   / (1024 * 1024))

    def measure(
        self,
        cache_dir: Path,
        host: HostSnapshot,
        disk_mb: int,
    ) -> CalibrationEntry:
        from supergraph.embedding.reranker import GGUFReranker
        from supergraph.config import SuperGraphConfig
        cfg = SuperGraphConfig()

        rss_baseline = _process_rss_mb()
        reranker = GGUFReranker(
            model_path=cfg.dsl.reranker_model_dir or "",
            projector_path=cfg.dsl.reranker_projector_path or "",
            n_gpu_layers=0,
        )
        rss_after_load = _process_rss_mb()
        ram_idle = max(rss_after_load - rss_baseline, 0)

        peak_rss = rss_after_load

        def _rerank_batch():
            nonlocal peak_rss
            scored = reranker.score(_PROBE_TEXT, [d for _, d in _PROBE_QUERY_DOC_PAIRS])
            peak_rss = max(peak_rss, _process_rss_mb())
            return len(scored) * 32

        tps = _measure_callable_tps(_rerank_batch, n_iters=2)
        ram_at_default = max(peak_rss - rss_baseline, 0)

        return CalibrationEntry(
            component_id=self.component_id,
            measured_at=datetime.now(timezone.utc).isoformat(),
            ram_mb_idle=ram_idle,
            ram_mb_at_default=ram_at_default,
            ram_mb_min=ram_idle,
            ram_mb_max=ram_at_default,
            disk_mb=disk_mb,
            vram_mb_full_offload=0,
            tps_cpu_threads={str(host.cpu_cores_logical): tps},
            tps_gpu_full_offload=None,
            extra={_EXTRA_RERANKER_MAX: 1024},
        )


class BonsaiProbe(Probe):

    def __init__(self, quant: str, skill: str):
        self.component_id = f"ingest:bonsai-{quant}-{skill}"
        self._quant = quant
        self._skill = skill

    def download(self, cache_dir: Path, host: HostSnapshot) -> int:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface-hub not installed; pip install 'supergraph[pro]'"
            ) from e
        repo = "superkaiii/Ternary-Bonsai-4B-GGUF"
        fname = f"Ternary-Bonsai-4B-{self._quant.upper()}.gguf"
        path = hf_hub_download(
            repo_id=repo, filename=fname,
            cache_dir=str(cache_dir / "models"),
        )
        return int(Path(path).stat().st_size / (1024 * 1024))

    def measure(
        self,
        cache_dir: Path,
        host: HostSnapshot,
        disk_mb: int,
    ) -> CalibrationEntry:
        from supergraph import SuperGraph
        from supergraph.bonsai_ingestor import (
            BonsaiIngestor,
            _DEFAULT_LITE_PROMPT_PATH, _DEFAULT_PROMPT_PATH,
        )
        skill_path = (_DEFAULT_LITE_PROMPT_PATH if self._skill == "lite"
                      else _DEFAULT_PROMPT_PATH)

        from huggingface_hub import scan_cache_dir
        gguf_path: Path | None = None
        repo_marker = "Ternary-Bonsai-4B-GGUF"
        file_marker = f"-{self._quant.upper()}.gguf"
        for repo in scan_cache_dir().repos:
            if repo_marker not in str(repo.repo_id):
                continue
            for rev in repo.revisions:
                for f in rev.files:
                    if f.file_name.endswith(file_marker):
                        gguf_path = Path(f.file_path)
                        break
        if gguf_path is None or not gguf_path.exists():
            raise RuntimeError(
                f"Bonsai GGUF not found after download for {self.component_id}; "
                "huggingface_hub cache may be corrupted - rerun "
                "`supergraph pro setup`."
            )

        rss_baseline = _process_rss_mb()

        gs = SuperGraph(embedder=None)
        try:
            ing = BonsaiIngestor(
                model_path=str(gguf_path),
                gs=gs,
                skill_path=str(skill_path),
                n_ctx=4096,
                n_gpu_layers=0,
                max_output_tokens=128,
                temperature=0.0,
            )
            ing.ingest("Probe warm-up.", msg_id="warm", dry_run=True)
            rss_after_load = _process_rss_mb()
            ram_idle = max(rss_after_load - rss_baseline, 0)

            peak_rss = rss_after_load

            def _measure_one():
                nonlocal peak_rss
                res = ing.ingest("Kailash joined OpenAI.",
                                msg_id="probe", dry_run=True)
                peak_rss = max(peak_rss, _process_rss_mb())
                try:
                    n_tok = len(ing._llm.tokenize(
                        res.raw_output.encode("utf-8"), add_bos=False,
                    ))
                except Exception:
                    n_tok = max(len(res.raw_output) // 4, 1)
                return n_tok

            tps_cpu = _measure_callable_tps(_measure_one, n_iters=3)
            ram_at_default = max(peak_rss - rss_baseline, 0)

            tps_gpu: float | None = None
            vram_full: int = 0
            if host.gpu_ready:
                ing._llm = None
                _free_caches()
                vram_pre = _vram_free_mb()
                try:
                    ing_gpu = BonsaiIngestor(
                        model_path=str(gguf_path),
                        gs=gs,
                        skill_path=str(skill_path),
                        n_ctx=4096,
                        n_gpu_layers=-1,
                        max_output_tokens=128,
                        temperature=0.0,
                    )
                    ing_gpu.ingest("Probe warm-up.", msg_id="warm", dry_run=True)
                    vram_post = _vram_free_mb()
                    vram_full = max(vram_pre - vram_post, 0)

                    def _gpu_one():
                        res = ing_gpu.ingest("Kailash joined OpenAI.",
                                            msg_id="probe", dry_run=True)
                        try:
                            return len(ing_gpu._llm.tokenize(
                                res.raw_output.encode("utf-8"), add_bos=False,
                            ))
                        except Exception:
                            return max(len(res.raw_output) // 4, 1)

                    tps_gpu = _measure_callable_tps(_gpu_one, n_iters=3)
                    ing_gpu._llm = None
                except Exception as e:
                    _log.warning("bonsai GPU probe failed: %s; CPU number stands", e)
        finally:
            gs.close()

        return CalibrationEntry(
            component_id=self.component_id,
            measured_at=datetime.now(timezone.utc).isoformat(),
            ram_mb_idle=ram_idle,
            ram_mb_at_default=ram_at_default,
            ram_mb_min=ram_idle,
            ram_mb_max=ram_at_default,
            disk_mb=disk_mb,
            vram_mb_full_offload=vram_full,
            tps_cpu_threads={str(host.cpu_cores_logical): tps_cpu},
            tps_gpu_full_offload=tps_gpu,
            extra={
                _EXTRA_N_CTX_MIN: 2048,
                _EXTRA_N_CTX_DEFAULT: 4096,
                _EXTRA_N_CTX_MAX: 8192,
                _EXTRA_N_BATCH: 512,
            },
        )


class WhisperProbe(Probe):

    def __init__(self, model_size: str):
        self.component_id = f"audio:whisper-{model_size}"
        self._model_size = model_size

    def download(self, cache_dir: Path, host: HostSnapshot) -> int:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper not installed; pip install 'supergraph[audio]'"
            ) from e
        WhisperModel(self._model_size, device="cpu", compute_type="int8")
        try:
            from huggingface_hub import scan_cache_dir
            for repo in scan_cache_dir().repos:
                if f"whisper-{self._model_size}" in str(repo.repo_id).lower():
                    return int(repo.size_on_disk / (1024 * 1024))
        except Exception:
            pass
        return 0

    def measure(
        self,
        cache_dir: Path,
        host: HostSnapshot,
        disk_mb: int,
    ) -> CalibrationEntry:
        from faster_whisper import WhisperModel
        sample = Path("/usr/share/sounds/alsa/Front_Center.wav")
        if not sample.exists():
            raise RuntimeError(
                "no probe WAV available at /usr/share/sounds/alsa/Front_Center.wav; "
                "supply a file via cache_dir/probe-audio.wav"
            )

        rss_baseline = _process_rss_mb()
        model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
        rss_after_load = _process_rss_mb()
        ram_idle = max(rss_after_load - rss_baseline, 0)
        peak_rss = rss_after_load

        def _one():
            nonlocal peak_rss
            segs, _info = model.transcribe(str(sample), beam_size=1)
            list(segs)
            peak_rss = max(peak_rss, _process_rss_mb())
            return 50

        tps = _measure_callable_tps(_one, n_iters=2)
        ram_at_default = max(peak_rss - rss_baseline, 0)

        return CalibrationEntry(
            component_id=self.component_id,
            measured_at=datetime.now(timezone.utc).isoformat(),
            ram_mb_idle=ram_idle,
            ram_mb_at_default=ram_at_default,
            ram_mb_min=ram_idle,
            ram_mb_max=ram_at_default,
            disk_mb=disk_mb,
            vram_mb_full_offload=0,
            tps_cpu_threads={str(host.cpu_cores_logical): tps},
            tps_gpu_full_offload=None,
            extra={},
        )


class VisionSidecarProbe(Probe):

    def __init__(self, model: str):
        self.component_id = f"vision:{model}"
        self._model = model

    def download(self, cache_dir: Path, host: HostSnapshot) -> int:
        try:
            from supergraph.ingest import vision_sidecar as vs
        except ImportError as e:
            raise RuntimeError(
                "vision sidecar deps missing; pip install 'supergraph[vision]'"
            ) from e
        model_path, mmproj_path = vs.download_weights(self._model)
        return int((model_path.stat().st_size + mmproj_path.stat().st_size) / (1024 * 1024))

    def measure(
        self,
        cache_dir: Path,
        host: HostSnapshot,
        disk_mb: int,
    ) -> CalibrationEntry:
        return CalibrationEntry(
            component_id=self.component_id,
            measured_at=datetime.now(timezone.utc).isoformat(),
            ram_mb_idle=0,
            ram_mb_at_default=0,
            ram_mb_min=0,
            ram_mb_max=0,
            disk_mb=disk_mb,
            vram_mb_full_offload=0,
            tps_cpu_threads={},
            tps_gpu_full_offload=None,
            extra={"placeholder": "vision RSS/VRAM measurement deferred to PR#3.6"},
        )


def _build_registry() -> dict[str, Callable[[], Probe]]:
    reg: dict[str, Callable[[], Probe]] = {
        "embedder:model2vec-256d":    lambda: Model2VecProbe(),
        "embedder:jina-v5-small":     lambda: JinaOnnxEmbedderProbe(
            "jina-v5-small", "jina-v5-small-retrieval", 1024,
        ),
        "embedder:jina-v5-nano":      lambda: JinaOnnxEmbedderProbe(
            "jina-v5-nano", "jina-v5-nano-retrieval", 768,
        ),
        "embedder:embeddinggemma-300m": lambda: JinaOnnxEmbedderProbe(
            "embeddinggemma-300m", "embeddinggemma-300m", 768,
        ),
        "embedder:fastembed-bge-small": lambda: FastembedProbe(),
        "reranker:jina-v3":           lambda: JinaV3RerankerProbe(),
        "ner:tinybert":               lambda: TinyBERTNERProbe(),
        "audio:whisper-tiny":         lambda: WhisperProbe("tiny"),
        "audio:whisper-base":         lambda: WhisperProbe("base"),
        "audio:whisper-small":        lambda: WhisperProbe("small"),
        "vision:smolvlm2-2.2b":       lambda: VisionSidecarProbe("smolvlm2-2.2b"),
        "vision:qwen-vl-3b":          lambda: VisionSidecarProbe("qwen-vl-3b"),
    }
    for q in ("tq1_0", "tq2_0"):
        for s in ("lite", "full"):
            cid = f"ingest:bonsai-{q}-{s}"
            reg[cid] = lambda q=q, s=s: BonsaiProbe(quant=q, skill=s)
    return reg


_REGISTRY: dict[str, Callable[[], Probe]] = _build_registry()


def list_probable() -> list[str]:
    return sorted(_REGISTRY.keys())


@dataclass
class ProbeRunSummary:

    successes: list[ProbeResult]
    failures: list[ProbeResult]
    total_duration_s: float

    @property
    def all_ok(self) -> bool:
        return not self.failures


def probe_components(
    component_ids: Iterable[str],
    host: HostSnapshot | None = None,
    cache_dir: Path | None = None,
    on_event: Callable[[str, dict], None] | None = None,
    skip_probe: bool = False,
) -> ProbeRunSummary:
    cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    host = host or HostSnapshot.capture(cache_dir=cache_dir)
    cache = CalibrationCache.load(host.host_signature(), cache_dir=cache_dir)

    successes: list[ProbeResult] = []
    failures: list[ProbeResult] = []
    t_total = time.perf_counter()

    for cid in component_ids:
        factory = _REGISTRY.get(cid)
        if factory is None:
            res = ProbeResult(
                component_id=cid, duration_s=0.0,
                error=f"no probe registered for {cid!r}",
            )
            failures.append(res)
            if on_event:
                on_event("probe_failed", {"component": cid, "error": res.error})
            continue

        if on_event:
            on_event("probe_start", {"component": cid})
        probe = factory()
        if skip_probe:
            t0 = time.perf_counter()
            try:
                disk_mb = probe.download(cache_dir, host)
                res = ProbeResult(
                    component_id=cid,
                    duration_s=time.perf_counter() - t0,
                    entry=CalibrationEntry(
                        component_id=cid,
                        measured_at=datetime.now(timezone.utc).isoformat(),
                        disk_mb=disk_mb,
                        extra={"download_only": True},
                    ),
                )
            except Exception as e:
                res = ProbeResult(
                    component_id=cid,
                    duration_s=time.perf_counter() - t0,
                    error=f"download failed: {type(e).__name__}: {e}",
                )
        else:
            res = probe.run(cache_dir, host)

        if res.error:
            failures.append(res)
            if on_event:
                on_event("probe_failed",
                         {"component": cid, "error": res.error,
                          "duration_s": res.duration_s})
        else:
            successes.append(res)
            if res.entry is not None:
                cache.components[cid] = res.entry
                cache.measured_at = datetime.now(timezone.utc).isoformat()
                cache.save(cache_dir=cache_dir)
            if on_event:
                on_event("probe_done",
                         {"component": cid, "duration_s": res.duration_s})

    return ProbeRunSummary(
        successes=successes,
        failures=failures,
        total_duration_s=time.perf_counter() - t_total,
    )
