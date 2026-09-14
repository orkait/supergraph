import json
import time

import pytest

from supergraph import SuperGraph

from superclaw.compaction import SUMMARY_LABEL
from supergraph.core.errors import SuperGraphError

from superclaw import maintain
from superclaw.dsl import now_ms
from superclaw.facts import Facts, as_of_ms, parse
from superclaw.memory import Memory, refusal
from superclaw.observations import ObservationStore
from superclaw.session import NAMESPACE, SessionStore
from superclaw.settings import LIMITS
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
    store.touch(a, "/ws/x.py", "read")
    store.touch(a, "/ws/x.py", "read")
    store.touch(a, "/ws/x.py", "wrote")
    store.touch(b, "/ws/y.py", "wrote")
    assert store.files_of(a) == [("/ws/x.py", "read"), ("/ws/x.py", "wrote")] and [s["id"] for s in store.touching("/ws/x.py")] == [a] and store.touching("/ws/none") == []
    ref = ObservationStore(gs).save(a, "read_file", "c9", "body " * 10)
    assert store.around(a)["produced"] == [f"obs:{ref}"] and store.around(a)["forked_from"] == [f"session:{b}"] and store.around(b) == {}
    hits = store.search("kept")
    assert hits and {h["id"] for h in hits} <= {a, b} and all(h["type"] == "message" for h in hits)
    assert any("kept" in h["text"] for h in hits) and store.search("zzzznomatch") == []
    store.append(a, "usage", {"input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 80, "cost_usd": 0.5})
    store.append(a, "usage", {"input_tokens": 30, "output_tokens": 10, "cost_usd": 0.25})
    assert store.usage(a) == {"calls": 2, "tokens": 160, "cached": 80, "cost_usd": 0.75} and store.usage(b) == {"calls": 0, "tokens": 0, "cached": 0, "cost_usd": 0.0}
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
    facts = mem.facts
    first = facts.assert_("Textual is built by Textualize.", "https://textual.textualize.io/", session_id="s1", observed_at=1_000_000)
    assert first.id.startswith("fact:") and facts.search("who builds Textual", as_of=1_000_001) == [first] and facts.search("who builds Textual", as_of=999_999) == []
    assert gs.execute(f'EDGES FROM "{first.id}"', namespace=NAMESPACE).data == []
    assert facts.assert_("textual  is built by Textualize.", "https://textual.textualize.io/").id == first.id and facts.recent()[0].observed_at > 1_000_000
    assert facts.render([first]) == f"- ({facts.render([first]).split(',')[0][3:]}, https://textual.textualize.io/) Textual is built by Textualize."
    assert first.id in mem.search_tool().run({"query": "who builds Textual"}, ctx).output and "textual.textualize.io" in mem.search_tool().run({"query": "who builds Textual"}, ctx).output
    assert note.run({"text": "Textual runs in the browser too.", "origin": "web", "source": "https://textual.textualize.io/"}, ctx).output.startswith("fact:")
    assert "source" in note.run({"text": "x", "origin": "web"}, ctx).output and "web fact" in refusal("x", "web")
    newer = facts.supersede(first.id, "Textual is built by Textualize.io Ltd.", "https://textual.textualize.io/about")
    believed = [f.id for f in facts.search("who builds Textual")]
    assert newer.id in believed and first.id not in believed and first.id not in [f.id for f in facts.recent()]
    assert parse('answer\n\nFacts:\n- Textual is a TUI framework | "a TUI framework"\n- Made by Textualize\n* starred line\nno dash') == [
        ("Textual is a TUI framework", '"a TUI framework"'), ("Made by Textualize", ""), ("starred line", "")] and parse("no heading") == []
    assert as_of_ms("2026-09-01") == 1_788_220_800_000 and as_of_ms("not a date") is None
    page = ObservationStore(gs).save("s1", "web_fetch", "", "the page about a niche zebra library " * 20)
    zebra = facts.assert_("Zebra renders stripes fast.", "https://zebra.example/", session_id="s1", page_ref=page, observed_at=5_000_000)
    sibling = facts.assert_("Zebra has no external dependencies.", "https://zebra.example/", session_id="s1", page_ref=page, observed_at=5_000_000)
    expanded = [f.id for f in facts.search("stripes")]
    assert expanded[0] == zebra.id and sibling.id in expanded and sibling.id not in [f.id for f in facts.search("stripes", limit=1)]
    assert sibling.id not in [f.id for f in facts.search("stripes", as_of=4_999_999)]
    reading = SuperGraph(embedder="none", enable_sentence_nodes=False, reader=lambda prompt: "Zebra, per the context.")
    told = Facts(reading)
    told.assert_("Zebra renders stripes fast.", "https://zebra.example/")
    answer = told.ask("what renders stripes")
    assert answer.text == "Zebra, per the context." and answer.cited and answer.cited[0].startswith("fact:")
    reading.close()
    with pytest.raises(SuperGraphError):
        facts.ask("who")
    assert maintain.stale(gs) and maintain.last(gs) is None
    day = 86_400_000
    old = facts.assert_("Old but solid.", "https://old.example/", observed_at=now_ms() - 100 * day)
    weak = facts.assert_("Old and weak.", "https://old.example/", confidence=0.3, observed_at=now_ms() - 100 * day)
    fresh = facts.assert_("Fresh.", "https://new.example/")
    gs.execute('CREATE NODE "obs:stale" kind = "obs" EXPIRES IN 1s DOCUMENT "stale page"', namespace=NAMESPACE)
    time.sleep(1.2)
    report = maintain.maintain(gs)
    assert (report.expired, report.decayed, report.retracted) == (1, 3, 1) and report.health.get("live_nodes") and "expired 1" in report.line("·")
    assert facts.load(zebra.id).confidence == 0.35
    assert facts.load(old.id).confidence == 0.35 and facts.load(weak.id) is None and facts.load(fresh.id).confidence == LIMITS.web_fact_confidence
    again = maintain.maintain(gs, optimize=False)
    assert (again.decayed, again.retracted, again.optimized) == (0, 0, {}) and not maintain.stale(gs) and maintain.last(gs)["expired"] == 0
    assert maintain.health_line(gs, "·").startswith("health tombstones ") and maintain.health_line(gs, "·").endswith("last maintain today") and maintain.health_line(None, "·").startswith("health not opened")
    assert maintain.snapshot(gs, "before") and not maintain.snapshot(gs, "before") and maintain.snapshots(gs) == ["before"]
    facts.retract(fresh.id, "testing rollback")
    assert facts.load(fresh.id) is None
    maintain.rollback(gs, "before")
    assert facts.load(fresh.id) is not None
    with pytest.raises(ValueError):
        facts.assert_("The key is sk-proj-abcdefghijklmnopqrstuvwxyz0123456789", "https://x.example/")
    assert not gs.execute('NODES WHERE kind = "memory"', namespace=NAMESPACE).data and gs.execute('NODES WHERE kind = "fact"', namespace=NAMESPACE).data
