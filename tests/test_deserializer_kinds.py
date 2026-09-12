import tempfile
from supergraph import SuperGraph


def test_kind_ids_above_255_survive_checkpoint():
    with tempfile.TemporaryDirectory() as td:
        gs = SuperGraph(path=td)
        for i in range(260):
            gs.execute(f'CREATE NODE "node_{i}" kind = "kind_{i}"')
        gs.checkpoint()
        gs.close()

        gs2 = SuperGraph(path=td)
        result = gs2.execute('NODE "node_259"')
        assert result.data is not None, "Node with kind > 255 not found after reload"
        assert result.data["kind"] == "kind_259", (
            f"Kind corrupted: expected 'kind_259', got '{result.data['kind']}'"
        )
        gs2.close()
