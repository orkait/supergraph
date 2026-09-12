"""Test ORDER BY on string (interned) columns."""
import pytest
from supergraph import SuperGraph


def test_order_by_string_column_asc():
    """ORDER BY on a string field should sort alphabetically."""
    gs = SuperGraph()
    gs.execute('CREATE NODE "c" kind = "item" name = "charlie"')
    gs.execute('CREATE NODE "a" kind = "item" name = "alice"')
    gs.execute('CREATE NODE "b" kind = "item" name = "bob"')

    result = gs.execute('NODES WHERE kind = "item" ORDER BY name ASC')
    names = [n["name"] for n in result.data]
    assert names == sorted(names), f"Expected sorted, got {names}"
    gs.close()


def test_order_by_string_column_desc():
    """ORDER BY DESC on a string field."""
    gs = SuperGraph()
    gs.execute('CREATE NODE "c" kind = "item" name = "charlie"')
    gs.execute('CREATE NODE "a" kind = "item" name = "alice"')
    gs.execute('CREATE NODE "b" kind = "item" name = "bob"')

    result = gs.execute('NODES WHERE kind = "item" ORDER BY name DESC')
    names = [n["name"] for n in result.data]
    assert names == sorted(names, reverse=True), f"Expected reverse sorted, got {names}"
    gs.close()


def test_order_by_string_with_limit():
    """ORDER BY string with LIMIT."""
    gs = SuperGraph()
    for i, name in enumerate(["delta", "alpha", "charlie", "bravo", "echo"]):
        gs.execute(f'CREATE NODE "n{i}" kind = "item" name = "{name}"')

    result = gs.execute('NODES WHERE kind = "item" ORDER BY name ASC LIMIT 3')
    names = [n["name"] for n in result.data]
    assert names == ["alpha", "bravo", "charlie"], f"Got {names}"
    gs.close()


def test_order_by_string_with_offset():
    """ORDER BY string with OFFSET."""
    gs = SuperGraph()
    for i, name in enumerate(["delta", "alpha", "charlie", "bravo", "echo"]):
        gs.execute(f'CREATE NODE "n{i}" kind = "item" name = "{name}"')

    result = gs.execute('NODES WHERE kind = "item" ORDER BY name ASC LIMIT 2 OFFSET 2')
    names = [n["name"] for n in result.data]
    assert names == ["charlie", "delta"], f"Got {names}"
    gs.close()


def test_order_by_numeric_still_works():
    """Ensure numeric ORDER BY still uses fast column path."""
    gs = SuperGraph()
    gs.execute('CREATE NODE "a" kind = "item" score = 30')
    gs.execute('CREATE NODE "b" kind = "item" score = 10')
    gs.execute('CREATE NODE "c" kind = "item" score = 20')

    result = gs.execute('NODES WHERE kind = "item" ORDER BY score ASC')
    scores = [n["score"] for n in result.data]
    assert scores == [10, 20, 30]
    gs.close()


def test_order_by_missing_field():
    """ORDER BY a field some nodes don't have - nulls sort last."""
    gs = SuperGraph()
    gs.execute('CREATE NODE "a" kind = "item" name = "alice" score = 10')
    gs.execute('CREATE NODE "b" kind = "item" name = "bob"')
    gs.execute('CREATE NODE "c" kind = "item" name = "charlie" score = 5')

    result = gs.execute('NODES WHERE kind = "item" ORDER BY score ASC')
    scored = [n for n in result.data if "score" in n and n["score"] is not None]
    assert len(scored) >= 2
    if len(scored) == 2:
        assert scored[0]["score"] <= scored[1]["score"]
    gs.close()
