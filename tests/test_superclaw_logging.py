import pytest

from supergraph import SuperGraph

from superclaw.loop import Options, run
from superclaw.policy import Mode, Policy
from superclaw.runtime import Completion, approx_tokens
from superclaw.session import SessionStore, prompt_hash
from superclaw.tools import Registry


class Scripted:
    def __init__(self, *completions):
        self.queue = list(completions)

    def complete(self, messages, tools):
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_prompt_is_logged_and_replay_ignores_log_only_events(tmp_path):
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    store = SessionStore(gs)
    sid = store.create(cwd=str(tmp_path), model="m")
    opts = Options(registry=Registry(), policy=Policy(tmp_path, Mode.ASK), workspace=tmp_path, system_prompt="SYS v1", session=store, session_id=sid)
    run("hi", Scripted(Completion(text="hello")), opts)
    assert [e["type"] for e in store.events(sid)] == ["prompt", "message", "message"]
    assert store.last_prompt(sid) == {"hash": prompt_hash("SYS v1"), "tokens": approx_tokens("SYS v1"), "text": "SYS v1"}
    assert [m.role for m in store.replay(sid)] == ["user", "assistant"]
    with pytest.raises(RuntimeError):
        run("again", Scripted(RuntimeError("provider down")), opts)
    assert store.events(sid)[-1]["type"] == "error"
    assert "provider down" in store.events(sid)[-1]["payload"]["error"]
    gs.close()
