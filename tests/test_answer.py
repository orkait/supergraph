"""ANSWER verb: retrieval + reader LLM synthesis.

supergraph ships no LLM dependency for ANSWER. Readers are plain
callables the user wires at SuperGraph construction. Tests use a
recording fake reader to verify the verb's glue without an LLM.
"""
from __future__ import annotations

import numpy as np
import pytest

from supergraph import SuperGraph, q
from supergraph.embedding.base import Embedder
from supergraph.core.errors import SuperGraphError


class FixedEmbedder(Embedder):
    @property
    def name(self): return "fixed"
    @property
    def dims(self): return 32

    def _encode(self, texts):
        vecs = []
        for t in texts:
            seed = hash(t) % (2**31)
            rng = np.random.RandomState(seed)
            v = rng.randn(32).astype(np.float32)
            v /= np.linalg.norm(v)
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

    def encode_documents(self, texts, titles=None):
        return self._encode(texts)

    def encode_queries(self, texts):
        return self._encode(texts)


class _Recorder:
    """Reader that captures every prompt and returns a scripted response."""
    def __init__(self, script: dict[str, str] | None = None, default: str = "[scripted]"):
        self.prompts: list[str] = []
        self.script = script or {}
        self.default = default

    def __call__(self, prompt: str, max_tokens: int = 1000) -> str:
        self.prompts.append(prompt)
        for needle, reply in self.script.items():
            if needle in prompt:
                return reply
        return self.default


# ---------- 1. Basic invocation ---------------------------------------------

def test_answer_end_to_end():
    """ANSWER retrieves via REMEMBER + calls the reader + returns answer shape."""
    rec = _Recorder({"capital of France": "Paris"})
    gs = SuperGraph(embedder=FixedEmbedder(), reader=rec)
    gs.execute('SYS REGISTER NODE KIND "m" REQUIRED content:string EMBED content')
    for i, t in enumerate([
        "Paris is the capital of France",
        "Rome is the capital of Italy",
        "Eiffel Tower is in Paris",
    ]):
        gs.execute(f'CREATE NODE "n{i}" kind = "m" content = "{t}"')

    r = gs.execute('ANSWER "What is the capital of France?" LIMIT 2')

    assert r.kind == "answer"
    assert r.count == 1
    assert r.data["answer"] == "Paris"
    assert r.data["cited_slots"]
    assert len(r.data["candidates"]) >= 1
    # Reader got called exactly once with a context-shaped prompt
    assert len(rec.prompts) == 1
    prompt = rec.prompts[0]
    assert "Context:" in prompt and "capital of France" in prompt

    # Meta carries the REMEMBER signals block
    sig = r.meta["signals"]
    assert sig["fusion"]["method"]
    assert sig["stages"]["final"] >= 1
    gs.close()


# ---------- 2. No reader configured -----------------------------------------

def test_answer_without_reader_raises():
    gs = SuperGraph(embedder=FixedEmbedder())
    gs.execute('SYS REGISTER NODE KIND "m" REQUIRED content:string EMBED content')
    gs.execute('CREATE NODE "n0" kind = "m" content = "Paris is the capital of France"')
    with pytest.raises(SuperGraphError, match="requires a configured reader"):
        gs.execute('ANSWER "anything" LIMIT 1')
    gs.close()


# ---------- 3. Named reader via USING ---------------------------------------

def test_answer_picks_named_reader_via_using():
    fast = _Recorder(default="fast-answer")
    careful = _Recorder(default="careful-answer")
    gs = SuperGraph(embedder=FixedEmbedder(), readers={"fast": fast, "careful": careful})
    gs.execute('SYS REGISTER NODE KIND "m" REQUIRED content:string EMBED content')
    gs.execute('CREATE NODE "n0" kind = "m" content = "Paris is the capital of France"')

    r1 = gs.execute('ANSWER "q" LIMIT 1 USING "fast"')
    r2 = gs.execute('ANSWER "q" LIMIT 1 USING "careful"')

    assert r1.data["answer"] == "fast-answer"
    assert r2.data["answer"] == "careful-answer"
    assert len(fast.prompts) == 1
    assert len(careful.prompts) == 1
    gs.close()


def test_answer_unknown_named_reader_raises():
    some = _Recorder()
    gs = SuperGraph(embedder=FixedEmbedder(), readers={"a": some})
    gs.execute('SYS REGISTER NODE KIND "m" REQUIRED content:string EMBED content')
    gs.execute('CREATE NODE "n0" kind = "m" content = "Paris"')
    with pytest.raises(SuperGraphError, match="named 'nope'"):
        gs.execute('ANSWER "q" LIMIT 1 USING "nope"')
    gs.close()


# ---------- 4. Reader exception handled without raise ------------------------

def test_answer_reader_exception_surfaced_in_result():
    def bad_reader(prompt, max_tokens=1000):
        raise RuntimeError("simulated api failure")
    gs = SuperGraph(embedder=FixedEmbedder(), reader=bad_reader)
    gs.execute('SYS REGISTER NODE KIND "m" REQUIRED content:string EMBED content')
    gs.execute('CREATE NODE "n0" kind = "m" content = "Paris is the capital of France"')

    r = gs.execute('ANSWER "q" LIMIT 1')
    assert r.data["answer"] == ""
    assert "error" in r.data
    assert "simulated api failure" in r.data["error"]
    # Retrieval still succeeded; candidates present
    assert len(r.data["candidates"]) >= 1
    gs.close()


# ---------- 5. Builder ------------------------------------------------------

def test_answer_builder_roundtrip_matches_string_dsl():
    rec = _Recorder({"Paris": "Paris"})
    gs = SuperGraph(embedder=FixedEmbedder(), reader=rec)
    gs.execute('SYS REGISTER NODE KIND "m" REQUIRED content:string EMBED content')
    gs.execute('CREATE NODE "n0" kind = "m" content = "Paris is the capital of France"')

    stmt = q.answer("What is the capital of France?", limit=3)
    assert stmt.dsl() == 'ANSWER "What is the capital of France?" LIMIT 3'
    r = stmt.execute(gs)
    assert r.kind == "answer"
    gs.close()


def test_answer_builder_compiles_full_surface():
    built = q.answer(
        "When did Caroline go to the support group?",
        limit=5,
        tokens=2000,
        at="2024-05",
        using="fast",
    )
    assert 'ANSWER "When did Caroline go to the support group?"' in built.dsl()
    assert 'AT "2024-05"' in built.dsl()
    assert "TOKENS 2000" in built.dsl()
    assert "LIMIT 5" in built.dsl()
    assert 'USING "fast"' in built.dsl()


# ---------- 6. Empty retrieval still answers (with no-context reply) ---------

def test_answer_on_empty_store_still_calls_reader():
    rec = _Recorder(default="no information available")
    gs = SuperGraph(embedder=FixedEmbedder(), reader=rec)
    gs.execute('SYS REGISTER NODE KIND "m" REQUIRED content:string EMBED content')
    # No nodes.
    r = gs.execute('ANSWER "anything" LIMIT 5')
    assert r.kind == "answer"
    assert r.data["answer"] == "no information available"
    assert r.data["candidates"] == []
    assert r.data["cited_slots"] == []
    # Prompt still contains a "(no retrieved context)" fallback
    assert "no retrieved context" in rec.prompts[0]
    gs.close()
