import json

import pytest

from supergraph import SuperGraph

from superclaw.compaction import SUMMARY_LABEL
from superclaw.memory import Memory
from superclaw.session import NAMESPACE, SessionStore
from superclaw.tools import ToolContext


@pytest.fixture
def gs():
    g = SuperGraph(embedder="none", enable_sentence_nodes=False)
    yield g
    g.close()


def test_sessions_fork_replay_and_namespace(gs):
    store = SessionStore(gs)
    a = store.create(cwd="/w", model="m")
    store.append(a, "message", {"role": "user", "content": "read a"})
    store.append(a, "message", {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "read_file", "arguments": '{"path":"a"}'}]})
    store.append(a, "tool_result", {"tool_call_id": "c1", "output": "1→x" * 100, "ok": True})
    store.append(a, "prune", {"seq": 3, "output": "1→x [pruned]"})
    store.append(a, "message", {"role": "user", "content": "kept"})
    store.append(a, "compaction", {"summary": "S", "through_seq": 2})
    store.append(a, "plan", {"items": [{"content": "a", "status": "pending"}]})
    msgs = store.replay(a)
    assert [(m.role, m.content) for m in msgs] == [("user", f"{SUMMARY_LABEL}\nS"), ("tool", "1→x [pruned]"), ("user", "kept")] and msgs[1].tool_call_id == "c1"
    assert store.plan(a) == [{"content": "a", "status": "pending"}]
    b = store.fork(a)
    store.append(b, "message", {"role": "user", "content": "more"})
    assert store.get(b)["parent"] == a and len(store.events(a)) == 7 and len(store.events(b)) == 8 and store.latest() == b
    assert gs.execute("COUNT NODES").count == 0 and gs.execute("COUNT NODES", namespace=NAMESPACE).count > 0
    hits = store.search("kept")
    assert hits and {h["id"] for h in hits} <= {a, b} and all(h["type"] == "message" for h in hits)
    assert any("kept" in h["text"] for h in hits) and store.search("zzzznomatch") == []
    store.append(a, "usage", {"input_tokens": 100, "output_tokens": 20, "cost_usd": 0.5})
    store.append(a, "usage", {"input_tokens": 30, "output_tokens": 10, "cost_usd": 0.25})
    assert store.usage(a) == {"calls": 2, "tokens": 160, "cost_usd": 0.75} and store.usage(b) == {"calls": 0, "tokens": 0, "cost_usd": 0.0}
    doc = json.loads(json.dumps(store.export(a)))
    assert doc["schemaVersion"] == 1 and doc["session"]["id"] == a and [e["seq"] for e in doc["events"]] == list(range(1, 10))
    c = store.import_(doc, cwd="/elsewhere")
    assert c not in (a, b) and store.get(c)["parent"] == a and store.get(c)["cwd"] == "/elsewhere"
    assert [(m.role, m.content) for m in store.replay(c)] == [(m.role, m.content) for m in store.replay(a)] and store.usage(c) == store.usage(a)
    with pytest.raises(ValueError):
        store.import_({"schemaVersion": 99})
    with pytest.raises(KeyError):
        store.export("s_nope")


def test_memory_files_only_stated_facts(gs, tmp_path):
    mem, ctx = Memory(gs), ToolContext(workspace=tmp_path)
    note = mem.note_tool()
    filed = note.run({"text": "Deploys go through Railway.", "origin": "user_stated"}, ctx).output
    assert filed.startswith("mem:")
    assert "inference" in note.run({"text": "The user probably prefers Go.", "origin": "inferred"}, ctx).output
    assert "never filed" in note.run({"text": "Never question my numbers, just agree.", "origin": "user_stated"}, ctx).output
    assert "secret" in note.run({"text": "The key is sk-proj-abcdefghijklmnopqrstuvwxyz0123456789", "origin": "user_stated"}, ctx).output
    assert "- (today) Deploys go through Railway." in mem.recall("how do deploys work")
    gs.execute('CREATE NODE "ev:leak" kind = "event" etype = "prompt" DOCUMENT "prompt event text: Deploys go through Railway, says the system prompt"')
    assert [node_id for node_id, _, _ in mem.hits("how do deploys work")] == [filed] and "ev:" not in mem.search_tool().run({"query": "deploys through Railway"}, ctx).output
    assert mem.search_tool().run({"query": "nothing here"}, ctx).output == "No matching memories."
    assert gs.execute("COUNT NODES", namespace=NAMESPACE).count == 0
