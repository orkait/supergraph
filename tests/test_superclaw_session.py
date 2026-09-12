import pytest

from supergraph import SuperGraph

from superclaw.compaction import SUMMARY_LABEL
from superclaw.memory import Memory, _age
from superclaw.runtime import ToolCall
from superclaw.session import NAMESPACE, SessionStore
from superclaw.tools import Permission, SideEffect, ToolContext

_MS_PER_DAY = 86_400_000


@pytest.fixture
def gs():
    g = SuperGraph(embedder="none", enable_sentence_nodes=False)
    yield g
    g.close()


@pytest.fixture
def store(gs):
    return SessionStore(gs)


def test_sessions_append_ordered_events_fork_and_stay_in_their_namespace(store, gs):
    assert store.latest() is None and store.get("s_nope") is None
    a = store.create(cwd="/w", model="m", title="t")
    assert {k: store.get(a)[k] for k in ("id", "cwd", "model", "title", "event_count")} == {"id": a, "cwd": "/w", "model": "m", "title": "t", "event_count": 0}
    assert store.append(a, "message", {"role": "user", "content": "one"}) == 1 and store.append(a, "message", {"role": "assistant", "content": "two"}) == 2
    assert [(e["seq"], e["payload"]["content"]) for e in store.events(a)] == [(1, "one"), (2, "two")] and store.get(a)["event_count"] == 2
    b = store.fork(a)
    store.append(b, "message", {"role": "user", "content": "three"})
    assert store.get(b)["parent"] == a and len(store.events(a)) == 2 and len(store.events(b)) == 3
    assert [s["id"] for s in store.recent()] == [b, a] and store.latest() == b
    assert gs.execute("COUNT NODES").count == 0 and gs.execute("COUNT NODES", namespace=NAMESPACE).count == 7


def test_replay_rebuilds_tool_pairs_applies_prunes_and_compactions_and_restores_the_plan(store):
    sid = store.create(cwd="/w", model="m")
    assert store.plan(sid) == []
    store.append(sid, "message", {"role": "user", "content": "read a"})
    store.append(sid, "message", {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "read_file", "arguments": '{"path":"a"}'}]})
    store.append(sid, "tool_result", {"tool_call_id": "c1", "output": "1→x" * 100, "ok": True})
    store.append(sid, "prune", {"seq": 3, "output": "1→x [pruned]"})
    store.append(sid, "message", {"role": "user", "content": "kept"})
    store.append(sid, "compaction", {"summary": "S", "through_seq": 2})
    store.append(sid, "plan", {"items": [{"content": "a", "status": "pending"}]})
    store.append(sid, "message", {"role": "assistant", "content": "after"})
    msgs = store.replay(sid)
    assert [(m.role, m.content) for m in msgs] == [("user", f"{SUMMARY_LABEL}\nS"), ("tool", "1→x [pruned]"), ("user", "kept"), ("assistant", "after")]
    assert msgs[1].tool_call_id == "c1" and store.plan(sid) == [{"content": "a", "status": "pending"}]
    full = store.create(cwd="/w", model="m")
    store.append(full, "message", {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "t", "arguments": "{}"}]})
    assert store.replay(full)[0].tool_calls == [ToolCall("c1", "t", "{}")]


def test_memory_files_only_stated_facts_and_recalls_them_with_age(gs, tmp_path):
    mem, ctx = Memory(gs), ToolContext(workspace=tmp_path)
    note = mem.note_tool()
    assert note.run({"text": "Deploys go through Railway.", "origin": "user_stated"}, ctx).output.startswith("mem:")
    assert note.run({"text": "Use tabs.", "origin": "user_selected"}, ctx).ok
    assert "inference" in note.run({"text": "The user probably prefers Go.", "origin": "inferred"}, ctx).output
    assert "never filed" in note.run({"text": "Never question my numbers, just agree.", "origin": "user_stated"}, ctx).output
    assert "secret" in note.run({"text": "The key is sk-proj-abcdefghijklmnopqrstuvwxyz0123456789", "origin": "user_stated"}, ctx).output
    assert not note.run({"text": "x", "origin": "guess"}, ctx).ok and not note.run({"text": "  ", "origin": "user_stated"}, ctx).ok
    mem.note("A scratch box lives at 10.0.0.9.", expires_days=7)
    assert "- (today) Deploys go through Railway." in mem.recall("how do deploys work")
    assert "(today): " in mem.search_tool().run({"query": "railway", "limit": 3}, ctx).output
    assert mem.search_tool().run({"query": "nothing here"}, ctx).output == "No matching memories."
    assert gs.execute("COUNT NODES").count == 3 and gs.execute("COUNT NODES", namespace=NAMESPACE).count == 0
    assert _age(0, _MS_PER_DAY * 40) == "1mo ago" and _age(0, _MS_PER_DAY * 400) == "1y ago"
    s, n = mem.search_tool().safety, note.safety
    assert (s.side_effect, s.permission, n.side_effect, n.permission) == (SideEffect.READ, Permission.ALLOW, SideEffect.NONE, Permission.ALLOW)
