import types
import pytest
from supergraph.llm_runner import LLMRunner


def _fake_completion_response(content: str):
    msg = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=msg)
    return types.SimpleNamespace(choices=[choice])


def _fake_stream_chunks(deltas):
    for d in deltas:
        delta = types.SimpleNamespace(content=d)
        choice = types.SimpleNamespace(delta=delta)
        yield types.SimpleNamespace(choices=[choice])


PROVIDERS = [{"pid": "groq/x", "litellm_model": "groq/x",
              "api_base": None, "api_key": "k"}]


def test_complete_messages_returns_first_nonempty(monkeypatch):
    import litellm
    calls = {}

    def fake_completion(**kw):
        calls["kw"] = kw
        return _fake_completion_response("UPSERT")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    runner = LLMRunner(PROVIDERS)
    out = runner.complete_messages(
        [{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.0,
    )
    assert out == "UPSERT"
    assert calls["kw"]["model"] == "groq/x"
    assert calls["kw"]["stream"] is False


def test_complete_messages_falls_back_on_error(monkeypatch):
    import litellm
    seq = iter([RuntimeError("boom"), _fake_completion_response("ok2")])

    def fake_completion(**kw):
        nxt = next(seq)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(litellm, "completion", fake_completion)
    runner = LLMRunner([
        {"pid": "a", "litellm_model": "groq/a", "api_base": None, "api_key": "k"},
        {"pid": "b", "litellm_model": "groq/b", "api_base": None, "api_key": "k"},
    ], retries=1)
    assert runner.complete_messages([{"role": "user", "content": "x"}]) == "ok2"


def test_stream_messages_yields_deltas(monkeypatch):
    import litellm

    def fake_completion(**kw):
        assert kw["stream"] is True
        return _fake_stream_chunks(["@U ", "alice ", "Alice"])

    monkeypatch.setattr(litellm, "completion", fake_completion)
    runner = LLMRunner(PROVIDERS)
    got = list(runner.stream_messages([{"role": "user", "content": "x"}]))
    assert "".join(got) == "@U alice Alice"


def test_synthesis_shim_reexports_bonsai_pipeline():
    from supergraph.ingest.llm import synthesis as S
    turn = S.parse_verb_output('@UPSERT alice Alice\n@FACT fav_color blue')
    assert ("alice", "Alice") in turn.entities
    assert ("fact:fav_color", "blue") in turn.beliefs
    dsl = S.synthesize_dsl(
        turn, msg_id="msg:1", session_id="s", role="user",
        text="alice likes blue", gs=None,
    )
    assert any(line.startswith('CREATE NODE "msg:1"') for line in dsl)
    assert any(line.startswith("ASSERT ") for line in dsl)
    assert S.IngestError is S.BonsaiError
    assert issubclass(S.IngestEmpty, S.IngestError)
    r = S.IngestResult(statements=["x"], executed=1, rejected=[],
                       entities_new=[], beliefs_changed=[], duration_ms=0)
    assert r.executed == 1


class _FakeRunner:
    def __init__(self, output="", deltas=None):
        self._output = output
        self._deltas = deltas or []
        self.last_model = "fake/model"

    def complete_messages(self, messages, *, max_tokens=1000, temperature=0.0):
        return self._output

    def stream_messages(self, messages, *, max_tokens=1000, temperature=0.0):
        for d in self._deltas:
            yield d


def _make_cloud(monkeypatch, gs, output="", deltas=None):
    from supergraph.ingest.llm import cloud as cloud_mod
    monkeypatch.setattr(
        cloud_mod, "build_provider_chain",
        lambda *a, **k: [{"pid": "fake", "litellm_model": "fake/m",
                          "api_base": None, "api_key": "k"}],
    )
    monkeypatch.setattr(
        cloud_mod, "LLMRunner",
        lambda chain, **kw: _FakeRunner(output=output, deltas=deltas),
    )
    return cloud_mod.CloudIngestor(gs=gs)


def test_cloud_batch_ingest_writes_graph(monkeypatch):
    from supergraph import SuperGraph
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    ci = _make_cloud(monkeypatch, gs, output="@UPSERT alice Alice\n@FACT fav blue")
    res = ci.ingest("alice likes blue", msg_id="msg:1")
    assert res.executed > 0
    assert not res.rejected
    assert gs.execute('NODE "msg:1"').count == 1


def test_cloud_batch_empty_output_raises(monkeypatch):
    from supergraph import SuperGraph
    from supergraph.ingest.llm.synthesis import IngestEmpty
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    ci = _make_cloud(monkeypatch, gs, output="<think>nothing</think>")
    with pytest.raises(IngestEmpty):
        ci.ingest("noop", msg_id="msg:2")


def test_cloud_batch_dry_run_does_not_write(monkeypatch):
    from supergraph import SuperGraph
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    ci = _make_cloud(monkeypatch, gs, output="@UPSERT bob Bob")
    res = ci.ingest("bob", msg_id="msg:3", dry_run=True)
    assert res.dry_run is True
    assert res.executed == 0
    assert gs.execute('NODE "msg:3"').count == 0


def test_cloud_stream_emits_phases_and_writes(monkeypatch):
    from supergraph import SuperGraph
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    ci = _make_cloud(monkeypatch, gs, deltas=["@UPSERT alice ", "Alice\n", "@FACT fav blue"])
    events = list(ci.ingest_stream("alice likes blue", msg_id="msg:s1"))
    phases = [e["phase"] for e in events]
    assert "generating" in phases
    assert "synthesizing" in phases
    assert "executing" in phases
    assert phases[-1] == "done"
    assert events[-1]["status"] == "ok"
    assert gs.execute('NODE "msg:s1"').count == 1
    assert any(e["phase"] == "executing" and e.get("status") == "ok" for e in events)


def test_cloud_stream_empty_output_done_empty(monkeypatch):
    from supergraph import SuperGraph
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    ci = _make_cloud(monkeypatch, gs, deltas=["<think>", "nothing", "</think>"])
    events = list(ci.ingest_stream("noop", msg_id="msg:s2"))
    assert events[-1] == {"phase": "done", "status": "empty"}
    assert gs.execute('NODE "msg:s2"').count == 0


def _patch_cloud(monkeypatch, output="", deltas=None):
    from supergraph.ingest.llm import cloud as cloud_mod
    monkeypatch.setattr(
        cloud_mod, "build_provider_chain",
        lambda *a, **k: [{"pid": "fake", "litellm_model": "fake/m",
                          "api_base": None, "api_key": "k"}],
    )
    monkeypatch.setattr(
        cloud_mod, "LLMRunner",
        lambda chain, **kw: _FakeRunner(output=output, deltas=deltas),
    )


def _cloud_gs(monkeypatch, output="", deltas=None):
    from supergraph import SuperGraph
    from supergraph.config import SuperGraphConfig, IngestConfig
    _patch_cloud(monkeypatch, output=output, deltas=deltas)
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    gs._config = SuperGraphConfig(ingest=IngestConfig(nl_backend="cloud"))
    return gs


def test_sg_ingest_nl_requires_backend():
    from supergraph import SuperGraph
    gs = SuperGraph(embedder="none")
    with pytest.raises(ValueError, match="nl_backend"):
        gs.ingest_nl("hello", msg_id="m1")


def test_sg_ingest_nl_cloud(monkeypatch):
    gs = _cloud_gs(monkeypatch, output="@UPSERT alice Alice")
    res = gs.ingest_nl("alice", msg_id="m2")
    assert res.executed > 0
    assert gs.execute('NODE "m2"').count == 1


def test_sg_ingest_nl_auto_msg_id(monkeypatch):
    gs = _cloud_gs(monkeypatch, output="@FACT mood good")
    res = gs.ingest_nl("i feel good")
    assert res.executed > 0


def test_sg_ingest_nl_stream_requires_cloud():
    from supergraph import SuperGraph
    gs = SuperGraph(embedder="none")
    with pytest.raises(ValueError, match="cloud"):
        list(gs.ingest_nl_stream("hi"))


def test_gs_cloud_auto_wires_answer_reader(monkeypatch):
    gs = _cloud_gs(monkeypatch, output="@UPSERT a A")
    assert gs._executor._reader is None
    gs.ingest_nl("seed", msg_id="m")
    assert callable(gs._executor._reader)
    assert gs._executor._reader("any prompt") == "@UPSERT a A"


def test_cloud_eager_wires_reader_at_construction(monkeypatch):
    from supergraph import SuperGraph
    _patch_cloud(monkeypatch, output="x")
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False, nl_backend="cloud")
    assert callable(gs._executor._reader)


def test_cloud_eager_no_key_is_graceful(monkeypatch):
    for k in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "CLOUDFLARE_API_KEY",
              "GOOGLE_AISTUDIO_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from supergraph import SuperGraph
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False, nl_backend="cloud")
    assert gs._executor._reader is None


def test_gs_cloud_respects_user_supplied_reader(monkeypatch):
    from supergraph import SuperGraph
    from supergraph.config import SuperGraphConfig, IngestConfig
    _patch_cloud(monkeypatch, output="@UPSERT a A")
    def sentinel(prompt, max_tokens=512):
        return "USER_READER"
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False, reader=sentinel)
    gs._config = SuperGraphConfig(ingest=IngestConfig(nl_backend="cloud"))
    gs.ingest_nl("seed", msg_id="m")
    assert gs._executor._reader is sentinel
