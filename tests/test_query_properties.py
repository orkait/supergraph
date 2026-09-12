from __future__ import annotations

import itertools


from supergraph.query.filters import F
from supergraph.query.escape import dsl_literal


_atoms = [
    F.eq("a", 1),
    F.eq("b", "x"),
    F.gt("c", 0.5),
    F.in_("d", [1, 2, 3]),
    F.like("e", "%x%"),
]


def test_and_commutative_semantic():
    for a, b in itertools.combinations(_atoms, 2):
        from supergraph.dsl.parser import parse
        parse(f"NODES WHERE {(a & b).to_dsl()}")
        parse(f"NODES WHERE {(b & a).to_dsl()}")


def test_or_commutative_semantic():
    for a, b in itertools.combinations(_atoms, 2):
        from supergraph.dsl.parser import parse
        parse(f"NODES WHERE {(a | b).to_dsl()}")
        parse(f"NODES WHERE {(b | a).to_dsl()}")


def test_and_associative():
    for a, b, c in itertools.combinations(_atoms, 3):
        assert ((a & b) & c).to_dsl() == (a & (b & c)).to_dsl()


def test_or_associative():
    for a, b, c in itertools.combinations(_atoms, 3):
        assert ((a | b) | c).to_dsl() == (a | (b | c)).to_dsl()


def test_double_negation():
    for a in _atoms:
        assert (~~a).to_dsl() == a.to_dsl()


def test_and_identity():
    for a in _atoms:
        assert (a & F.true()).to_dsl() == a.to_dsl()
        assert (F.true() & a).to_dsl() == a.to_dsl()


def test_or_identity():
    for a in _atoms:
        assert (a | F.false()).to_dsl() == a.to_dsl()
        assert (F.false() | a).to_dsl() == a.to_dsl()


def test_and_absorption_false():
    for a in _atoms:
        assert (a & F.false()).to_dsl() == "false"


def test_or_absorption_true():
    for a in _atoms:
        assert (a | F.true()).to_dsl() == "true"


ADVERSARIAL_STRINGS = [
    'simple',
    'with "quotes" inside',
    r'with \ backslash',
    r'end with \ ',
    '"; DROP ALL; --',
    '--comment',
    '\\',
    '\\\\',
    '\\"',
    '""""',
    '\n newline \n',
    '\x00 null byte',
    '\t tab',
    "'single quotes'",
    "unicode é ü 汉字 🔥",
]


def test_every_adversarial_string_emits_balanced_quotes():
    from supergraph.dsl.parser import parse
    for s in ADVERSARIAL_STRINGS:
        out = dsl_literal(s)
        assert out.startswith('"')
        assert out.endswith('"')
        parse(f'NODES WHERE kind = {out}')


def test_every_adversarial_string_in_F_eq():
    from supergraph.dsl.parser import parse
    for s in ADVERSARIAL_STRINGS:
        f = F.eq("kind", s)
        parse(f"NODES WHERE {f.to_dsl()}")


def test_every_adversarial_string_in_document_clause():
    from supergraph import q
    from supergraph.dsl.parser import parse
    for s in ADVERSARIAL_STRINGS:
        dsl = q.create_node("m1", kind="memory", document=s).dsl()
        parse(dsl)


def test_every_value_type_round_trips_in_where():
    from datetime import date, datetime
    from supergraph.dsl.parser import parse
    values = [
        "str",
        42,
        -7,
        0,
        0.5,
        -0.5,
        True,
        False,
        None,
        date(2024, 3, 15),
        datetime(2024, 3, 15, 10, 30, 0),
    ]
    for v in values:
        f = F.eq("x", v) if v is not None else F.is_null("x")
        parse(f"NODES WHERE {f.to_dsl()}")


def test_deeply_nested_and():
    from supergraph.dsl.parser import parse
    f = F.eq("a", 0)
    for i in range(1, 20):
        f = f & F.eq(f"a{i}", i)
    dsl = f.to_dsl()
    parse(f"NODES WHERE {dsl}")


def test_deeply_nested_or():
    from supergraph.dsl.parser import parse
    f = F.eq("a", 0)
    for i in range(1, 20):
        f = f | F.eq(f"a{i}", i)
    dsl = f.to_dsl()
    parse(f"NODES WHERE {dsl}")


def test_mixed_nesting():
    from supergraph.dsl.parser import parse
    f = (F.eq("a", 1) & F.gt("b", 0)) | (F.eq("c", "x") & ~F.eq("d", True))
    parse(f"NODES WHERE {f.to_dsl()}")
