"""Walk/path anchor resolution in @-verb synthesis (cloud + local share this)."""
from supergraph import SuperGraph
from supergraph.ingest.llm import synthesis as S


def _synth(gs, verb_line, msg_id):
    turn = S.parse_verb_output(verb_line)
    return S.synthesize_dsl(turn, msg_id=msg_id, session_id="s", role="user",
                            text="q", gs=gs)


def test_walk_anchor_no_double_prefix_when_unknown():
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    stmts = _synth(gs, "@RECALL ent:unknown_person", "m1")
    recall = next(s for s in stmts if s.startswith("RECALL"))
    assert "ent:ent:" not in recall                 # bug was ent:ent:
    assert '"ent:unknown_person"' in recall          # single prefix fallback


def test_walk_anchor_resolves_cross_turn_entity():
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    # entity created in a prior turn (hashed id), with a canonical_name
    gs.execute('CREATE NODE "entity:abc123" kind = "entity" canonical_name = "Marie Curie"')
    stmts = _synth(gs, "@RECALL ent:marie_curie", "m2")
    recall = next(s for s in stmts if s.startswith("RECALL"))
    assert "ent:ent:" not in recall
    assert "entity:abc123" in recall                 # resolved slug -> real hashed id


def test_path_pair_resolves_both_anchors_cross_turn():
    gs = SuperGraph(embedder="none", enable_sentence_nodes=False)
    gs.execute('CREATE NODE "entity:aaa" kind = "entity" canonical_name = "Marie Curie"')
    gs.execute('CREATE NODE "entity:bbb" kind = "entity" canonical_name = "Pierre Curie"')
    stmts = _synth(gs, "@PATH ent:marie_curie ent:pierre_curie", "m3")
    path = next(s for s in stmts if s.startswith("PATH"))
    assert "ent:ent:" not in path
    assert "entity:aaa" in path and "entity:bbb" in path


def test_search_aliases_to_lexical():
    # models frequently emit @SEARCH; alias it to @LEXICAL instead of dropping
    turn = S.parse_verb_output("@SEARCH radium isotopes")
    assert any(s.startswith('LEXICAL SEARCH "radium isotopes"') for s in turn.statements)
