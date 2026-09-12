import pytest

from supergraph import SuperGraph

from superclaw.compaction import SUMMARY_LABEL
from superclaw.runtime import ToolCall
from superclaw.session import NAMESPACE, SessionStore


@pytest.fixture
def gs():
    g = SuperGraph(embedder="none", enable_sentence_nodes=False)
    yield g
    g.close()


@pytest.fixture
def store(gs):
    return SessionStore(gs)


class TestLifecycle:
    def test_create_and_get(self, store):
        sid = store.create(cwd="/w", model="m", title="t")
        meta = store.get(sid)
        assert meta["id"] == sid
        assert meta["cwd"] == "/w"
        assert meta["model"] == "m"
        assert meta["title"] == "t"
        assert meta["event_count"] == 0

    def test_get_missing_is_none(self, store):
        assert store.get("s_nope") is None

    def test_append_returns_increasing_seq_and_events_are_ordered(self, store):
        sid = store.create(cwd="/w", model="m")
        assert store.append(sid, "message", {"role": "user", "content": "hi"}) == 1
        assert store.append(sid, "message", {"role": "assistant", "content": "yo"}) == 2
        evs = store.events(sid)
        assert [(e["seq"], e["type"], e["payload"]["content"]) for e in evs] == [(1, "message", "hi"), (2, "message", "yo")]
        assert store.get(sid)["event_count"] == 2

    def test_list_newest_first_and_latest(self, store):
        a = store.create(cwd="/w", model="m")
        b = store.create(cwd="/w", model="m")
        assert [s["id"] for s in store.list()] == [b, a]
        assert store.latest() == b

    def test_latest_is_none_when_empty(self, store):
        assert store.latest() is None

    def test_fork_copies_events_and_records_parent(self, store):
        a = store.create(cwd="/w", model="m")
        store.append(a, "message", {"role": "user", "content": "one"})
        b = store.fork(a)
        assert b != a
        assert [e["payload"]["content"] for e in store.events(b)] == ["one"]
        assert store.get(b)["parent"] == a
        store.append(b, "message", {"role": "user", "content": "two"})
        assert len(store.events(a)) == 1

    def test_writes_stay_in_the_superclaw_namespace(self, store, gs):
        sid = store.create(cwd="/w", model="m")
        store.append(sid, "message", {"role": "user", "content": "hi"})
        assert gs.execute("COUNT NODES").count == 0
        assert gs.execute("COUNT NODES", namespace=NAMESPACE).count == 2


class TestReplay:
    def test_rebuilds_messages_with_tool_pairs(self, store):
        sid = store.create(cwd="/w", model="m")
        store.append(sid, "message", {"role": "user", "content": "read a"})
        store.append(sid, "message", {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "read_file", "arguments": '{"path":"a"}'}]})
        store.append(sid, "tool_result", {"tool_call_id": "c1", "name": "read_file", "output": "1→x", "ok": True})
        store.append(sid, "message", {"role": "assistant", "content": "done"})
        msgs = store.replay(sid)
        assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]
        assert msgs[1].tool_calls == [ToolCall("c1", "read_file", '{"path":"a"}')]
        assert msgs[2].tool_call_id == "c1" and msgs[2].content == "1→x" and not msgs[2].is_error

    def test_compaction_event_replaces_earlier_messages(self, store):
        sid = store.create(cwd="/w", model="m")
        store.append(sid, "message", {"role": "user", "content": "old1"})
        store.append(sid, "message", {"role": "assistant", "content": "old2"})
        store.append(sid, "message", {"role": "user", "content": "kept"})
        store.append(sid, "compaction", {"summary": "S", "through_seq": 2})
        store.append(sid, "message", {"role": "assistant", "content": "after"})
        msgs = store.replay(sid)
        assert [m.content for m in msgs] == [f"{SUMMARY_LABEL}\nS", "kept", "after"]
        assert msgs[0].role == "user"

    def test_plan_event_restores_plan_state(self, store):
        sid = store.create(cwd="/w", model="m")
        store.append(sid, "plan", {"items": [{"content": "a", "status": "pending"}]})
        assert store.plan(sid) == [{"content": "a", "status": "pending"}]

    def test_plan_is_empty_without_events(self, store):
        sid = store.create(cwd="/w", model="m")
        assert store.plan(sid) == []
