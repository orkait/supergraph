from supergraph.core.store import CoreStore


def test_get_edges_to_uses_transpose():
    store = CoreStore()
    store.put_node("hub", "node", {})
    for i in range(100):
        store.put_node(f"spoke_{i}", "node", {})
        store.put_edge(f"spoke_{i}", "hub", "points_to")

    edges = store.get_edges_to("hub", kind="points_to")
    assert len(edges) == 100
    sources = {e["source"] for e in edges}
    assert sources == {f"spoke_{i}" for i in range(100)}


def test_get_edges_to_multiple_types():
    store = CoreStore()
    store.put_node("target", "node", {})
    store.put_node("a", "node", {})
    store.put_node("b", "node", {})
    store.put_edge("a", "target", "likes")
    store.put_edge("b", "target", "follows")

    edges = store.get_edges_to("target")
    assert len(edges) == 2
    kinds = {e["kind"] for e in edges}
    assert kinds == {"likes", "follows"}
