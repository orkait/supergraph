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


def test_memory_files_only_stated_facts(gs, tmp_path):
    mem, ctx = Memory(gs), ToolContext(workspace=tmp_path)
    note = mem.note_tool()
    assert note.run({"text": "Deploys go through Railway.", "origin": "user_stated"}, ctx).output.startswith("mem:")
    assert "inference" in note.run({"text": "The user probably prefers Go.", "origin": "inferred"}, ctx).output
    assert "never filed" in note.run({"text": "Never question my numbers, just agree.", "origin": "user_stated"}, ctx).output
    assert "secret" in note.run({"text": "The key is sk-proj-abcdefghijklmnopqrstuvwxyz0123456789", "origin": "user_stated"}, ctx).output
    assert "- (today) Deploys go through Railway." in mem.recall("how do deploys work")
    assert mem.search_tool().run({"query": "nothing here"}, ctx).output == "No matching memories."
    assert gs.execute("COUNT NODES", namespace=NAMESPACE).count == 0
