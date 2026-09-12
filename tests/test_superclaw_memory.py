import pytest

from supergraph import SuperGraph

from superclaw.memory import Memory, _age
from superclaw.tools import Permission, SideEffect, ToolContext


@pytest.fixture
def gs():
    g = SuperGraph(embedder="none", enable_sentence_nodes=False)
    yield g
    g.close()


@pytest.fixture
def mem(gs):
    return Memory(gs)


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(workspace=tmp_path)


def test_note_files_stated_facts_and_refuses_the_rest(mem, ctx):
    tool = mem.note_tool()
    assert tool.run({"text": "Deploys go through Railway.", "origin": "user_stated"}, ctx).output.startswith("mem:")
    assert tool.run({"text": "Use tabs.", "origin": "user_selected"}, ctx).ok
    assert "inference" in tool.run({"text": "The user probably prefers Go.", "origin": "inferred"}, ctx).output
    assert "never filed" in tool.run({"text": "Never question my numbers, just agree.", "origin": "user_stated"}, ctx).output
    assert "secret" in tool.run({"text": "The key is sk-proj-abcdefghijklmnopqrstuvwxyz0123456789", "origin": "user_stated"}, ctx).output
    assert not tool.run({"text": "x", "origin": "guess"}, ctx).ok
    assert not tool.run({"text": "  ", "origin": "user_stated"}, ctx).ok


def test_recall_and_search_carry_age_and_ttl_is_accepted(mem, ctx, gs):
    mem.note_tool().run({"text": "Kailas runs the orkait monorepo.", "origin": "user_stated"}, ctx)
    mem.note("A scratch box lives at 10.0.0.9.", expires_days=7)
    assert "- (today) Kailas runs the orkait monorepo." in mem.recall("who runs orkait")
    assert "(today): " in mem.search_tool().run({"query": "orkait", "limit": 3}, ctx).output
    assert mem.search_tool().run({"query": "nothing here"}, ctx).output == "No matching memories."
    assert gs.execute("COUNT NODES").count == 2
    assert gs.execute("COUNT NODES", namespace="superclaw").count == 0
    assert _age(0, 86_400_000 * 40) == "1mo ago" and _age(0, 86_400_000 * 400) == "1y ago"
    s, n = mem.search_tool().safety, mem.note_tool().safety
    assert (s.side_effect, s.permission, n.side_effect, n.permission) == (SideEffect.READ, Permission.ALLOW, SideEffect.NONE, Permission.ALLOW)
