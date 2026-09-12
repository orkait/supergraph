"""Unit tests for bonsai_ingestor correctness guards.

The live LLM path needs the 1.09 GB TQ1_0 GGUF on disk and is skipped by
default. These tests exercise the post-processing and guard logic without
touching llama.cpp.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from supergraph.bonsai_ingestor import (
    BonsaiIngestor,
    ParsedTurn,
    FactState,
    IngestEmpty,
    IngestOverflow,
    IngestResult,
    _dedupe_upserts,
    _dsl_escape,
    _parse_verb_output,
    _render_known_facts_block,
    _scrape_belief_updates,
    _split_lines,
    _strip_think,
    _synthesize_dsl,
)


# --------------------------------------------------------------------
# Post-processing helpers
# --------------------------------------------------------------------

def test_strip_think_removes_single_block():
    out = _strip_think("<think>reasoning</think>CREATE NODE \"x\" kind = \"k\"")
    assert "think" not in out.lower()
    assert "CREATE NODE" in out


def test_strip_think_removes_multiple_blocks():
    inp = "<think>a</think>line1<think>b</think>line2"
    assert _strip_think(inp) == "line1line2"


def test_strip_think_empty_on_only_think():
    assert _strip_think("<think>foo</think>") == ""


def test_split_lines_drops_fences_and_blanks():
    inp = "```dsl\nCREATE NODE \"a\" kind = \"k\"\n\n```\nCREATE NODE \"b\" kind = \"k\""
    lines = _split_lines(inp)
    assert lines == [
        'CREATE NODE "a" kind = "k"',
        'CREATE NODE "b" kind = "k"',
    ]


def test_dedupe_upserts_keeps_first_drops_later():
    stmts = [
        'UPSERT NODE "ent:priya" kind = "entity" name = "Priya"',
        'UPSERT NODE "ent:openai" kind = "entity" name = "OpenAI"',
        'UPSERT NODE "ent:priya" kind = "entity" name = "Priya"',
        'CREATE EDGE "m:0" -> "ent:priya" kind = "mentions"',
    ]
    kept, dropped = _dedupe_upserts(stmts)
    assert len(kept) == 3
    assert 'ent:priya' in kept[0]
    assert 'ent:openai' in kept[1]
    assert 'CREATE EDGE' in kept[2]
    assert len(dropped) == 1
    assert "duplicate" in dropped[0][1]


def test_dedupe_upserts_passes_non_upsert():
    stmts = ['CREATE NODE "m:0" kind = "message"', 'CREATE EDGE "a" -> "b" kind = "k"']
    kept, dropped = _dedupe_upserts(stmts)
    assert kept == stmts
    assert dropped == []


# --------------------------------------------------------------------
# Skill fingerprint
# --------------------------------------------------------------------

def test_skill_fingerprint_is_stable_across_instances(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("content A")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"ignored; no llama init yet")

    ing1 = BonsaiIngestor(model_path=model, skill_path=skill)
    ing2 = BonsaiIngestor(model_path=model, skill_path=skill)
    assert ing1.skill_fingerprint == ing2.skill_fingerprint

    expected = hashlib.sha256(b"content A").hexdigest()[:12]
    assert ing1.skill_fingerprint == expected


def test_skill_fingerprint_changes_on_skill_edit(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("v1")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"")

    ing = BonsaiIngestor(model_path=model, skill_path=skill)
    fp_v1 = ing.skill_fingerprint

    skill.write_text("v2")
    ing._reload_skill()
    assert ing.skill_fingerprint != fp_v1


def test_skill_fingerprint_pinned_into_system_prompt(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("rule body")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"")

    ing = BonsaiIngestor(model_path=model, skill_path=skill)
    assert f"skill-sha256={ing.skill_fingerprint}" in ing._system_prompt
    assert "rule body" in ing._system_prompt


# --------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------

def test_empty_input_raises_ingest_empty(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("any")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"")

    ing = BonsaiIngestor(model_path=model, skill_path=skill)
    with pytest.raises(IngestEmpty):
        ing.ingest("")
    with pytest.raises(IngestEmpty):
        ing.ingest("   \n\t  ")


def test_non_dry_run_without_store_raises(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("any")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"")

    ing = BonsaiIngestor(model_path=model, skill_path=skill)
    with pytest.raises(ValueError, match="requires a SuperGraph"):
        ing.ingest("hello")


def test_missing_model_file_raises(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("any")
    with pytest.raises(FileNotFoundError):
        BonsaiIngestor(model_path=tmp_path / "nope.gguf", skill_path=skill)


def test_missing_skill_file_raises(tmp_path: Path):
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        BonsaiIngestor(model_path=model, skill_path=tmp_path / "nope.md")


# --------------------------------------------------------------------
# Frontmatter strip
# --------------------------------------------------------------------

def test_yaml_frontmatter_stripped_from_skill(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("---\nname: x\n---\n\nactual rules")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"")

    ing = BonsaiIngestor(model_path=model, skill_path=skill)
    assert "name: x" not in ing._skill_text
    assert "actual rules" in ing._skill_text


# --------------------------------------------------------------------
# IngestResult shape
# --------------------------------------------------------------------

def test_ingest_result_defaults():
    r = IngestResult()
    assert r.statements == []
    assert r.executed == 0
    assert r.rejected == []
    assert r.entities_new == []
    assert r.beliefs_changed == []
    assert r.duration_ms == 0
    assert r.raw_output == ""
    assert r.skill_fingerprint == ""
    assert r.dry_run is False


# --------------------------------------------------------------------
# Fact state tracking (cross-message belief identity)
# --------------------------------------------------------------------

def test_scrape_single_assert_creates_factstate():
    facts: dict[str, FactState] = {}
    _scrape_belief_updates(
        ['ASSERT "fact:color" kind = "belief" value = "blue" CONFIDENCE 0.9 SOURCE "m:0"'],
        facts,
    )
    assert "fact:color" in facts
    st = facts["fact:color"]
    assert st.kind == "belief"
    assert st.value == "blue"
    assert st.confidence == 0.9
    assert st.source == "m:0"
    assert st.retracted is False


def test_scrape_assert_then_retract_marks_retracted():
    facts: dict[str, FactState] = {}
    _scrape_belief_updates(
        ['ASSERT "fact:x" kind = "belief" value = "v" CONFIDENCE 0.9 SOURCE "m:0"'],
        facts,
    )
    _scrape_belief_updates(
        ['RETRACT "fact:x" REASON "wrong"'],
        facts,
    )
    assert facts["fact:x"].retracted is True
    assert facts["fact:x"].retract_reason == "wrong"


def test_scrape_retract_then_reassert_un_retracts():
    facts: dict[str, FactState] = {}
    _scrape_belief_updates(
        ['ASSERT "fact:x" kind = "belief" value = "old" CONFIDENCE 0.9 SOURCE "m:0"',
         'RETRACT "fact:x" REASON "wrong"',
         'ASSERT "fact:x" kind = "belief" value = "new" CONFIDENCE 0.9 SOURCE "m:1"'],
        facts,
    )
    st = facts["fact:x"]
    assert st.value == "new"
    assert st.retracted is False
    assert st.retract_reason == ""


def test_scrape_ignores_non_belief_lines():
    facts: dict[str, FactState] = {}
    _scrape_belief_updates(
        ['CREATE NODE "m:0" kind = "message"',
         'UPSERT NODE "ent:x" kind = "entity" name = "X"',
         'CREATE EDGE "a" -> "b" kind = "mentions"'],
        facts,
    )
    assert facts == {}


def test_render_known_facts_block_empty_when_no_facts():
    assert _render_known_facts_block({}) == ""


def test_render_known_facts_block_hides_retracted():
    facts = {
        "fact:a": FactState(fact_id="fact:a", kind="belief", value="alive", confidence=0.9),
        "fact:b": FactState(fact_id="fact:b", kind="belief", value="dead", retracted=True),
    }
    block = _render_known_facts_block(facts)
    assert "fact:a" in block
    assert "fact:b" not in block
    assert "alive" in block
    assert "dead" not in block


def test_render_known_facts_block_formats_each_fact():
    facts = {
        "fact:color": FactState(
            fact_id="fact:color",
            kind="belief",
            value="blue",
            confidence=0.9,
            source="m:s1:0",
        ),
    }
    block = _render_known_facts_block(facts)
    assert "[fact:color]" in block
    assert 'kind="belief"' in block
    assert 'value="blue"' in block
    assert "confidence=0.90" in block
    assert 'source="m:s1:0"' in block
    assert "KNOWN FACTS" in block


def test_render_known_facts_trims_to_max_facts():
    facts = {
        f"fact:{i}": FactState(fact_id=f"fact:{i}", value=str(i), confidence=0.9)
        for i in range(60)
    }
    block = _render_known_facts_block(facts, max_facts=5)
    for i in range(55, 60):
        assert f"[fact:{i}]" in block
    for i in range(0, 55):
        assert f"[fact:{i}]" not in block


def test_ingestor_facts_property_returns_copy(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("any")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"")

    ing = BonsaiIngestor(model_path=model, skill_path=skill)
    assert ing.facts == {}
    # Simulate state set by an earlier ingest:
    ing._facts["fact:x"] = FactState(fact_id="fact:x", value="v")
    snapshot = ing.facts
    assert "fact:x" in snapshot
    # Mutating the snapshot should not affect internal state
    snapshot["fact:y"] = FactState(fact_id="fact:y")
    assert "fact:y" not in ing._facts


def test_ingestor_reset_facts_clears_state(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("any")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"")

    ing = BonsaiIngestor(model_path=model, skill_path=skill)
    ing._facts["fact:x"] = FactState(fact_id="fact:x", value="v")
    ing.reset_facts()
    assert ing._facts == {}


# --------------------------------------------------------------------
# Verb parser (English-keyword @-prefix grammar)
# --------------------------------------------------------------------

def test_parse_all_three_ingest_verbs():
    out = '''@UPSERT priya Priya
@UPSERT openai OpenAI
@BELIEF color blue
@RETRACT old'''
    turn = _parse_verb_output(out)
    assert turn.entities == [("priya", "Priya"), ("openai", "OpenAI")]
    assert turn.beliefs == [("fact:color", "blue")]
    assert turn.retracts == ["fact:old"]


def test_parse_empty_output_is_empty_turn():
    turn = _parse_verb_output("")
    assert turn.entities == []
    assert turn.beliefs == []
    assert turn.retracts == []


def test_parse_entities_only():
    turn = _parse_verb_output("@UPSERT kailash Kailash")
    assert turn.entities == [("kailash", "Kailash")]
    assert turn.beliefs == []
    assert turn.retracts == []


def test_parse_multi_word_name_joined_by_whitespace():
    """Rest-of-line is the name; split on first 2 whitespace runs only."""
    turn = _parse_verb_output("@UPSERT sf San Francisco")
    assert turn.entities == [("sf", "San Francisco")]


def test_parse_case_insensitive_verbs():
    out = "@upsert priya Priya\n@belief color blue\n@retract old"
    turn = _parse_verb_output(out)
    assert turn.entities == [("priya", "Priya")]
    assert turn.beliefs == [("fact:color", "blue")]
    assert turn.retracts == ["fact:old"]


def test_parse_assert_alias_maps_to_belief():
    """ASSERT is a grammar keyword; we accept it as an alias for @BELIEF."""
    turn = _parse_verb_output("@ASSERT color blue")
    assert turn.beliefs == [("fact:color", "blue")]


def test_parse_tolerates_fence_lines():
    out = "```\n@UPSERT x X\n```"
    turn = _parse_verb_output(out)
    assert turn.entities == [("x", "X")]


def test_parse_strips_prefix_if_model_adds_it():
    """Model sometimes emits '@UPSERT ent:x X'; we normalize to slug-only."""
    turn = _parse_verb_output('@UPSERT ent:priya Priya')
    assert turn.entities == [("priya", "Priya")]

    turn2 = _parse_verb_output('@BELIEF fact:color blue')
    assert turn2.beliefs == [("fact:color", "blue")]


def test_parse_ignores_unknown_verbs():
    out = "@UPSERT priya Priya\n@FOO some garbage\n@RETRACT old"
    turn = _parse_verb_output(out)
    assert turn.entities == [("priya", "Priya")]
    assert turn.retracts == ["fact:old"]


def test_parse_ignores_malformed_short_lines():
    """Missing required args -> line dropped, no crash."""
    out = "@UPSERT justslug\n@BELIEF onlytopic\n@RETRACT\n"
    turn = _parse_verb_output(out)
    assert turn.entities == []
    assert turn.beliefs == []
    assert turn.retracts == []


def test_parse_strips_quotes_if_present():
    """Model occasionally wraps tokens in quotes; handle both."""
    turn = _parse_verb_output('@UPSERT "priya" "Priya"')
    assert turn.entities == [("priya", "Priya")]


# --------------------------------------------------------------------
# @-prefix contract
# --------------------------------------------------------------------

def test_parse_drops_lines_without_at_prefix():
    """Any line not starting with @ drops silently (English drift inert)."""
    out = '''UPSERT priya Priya
Wait, let me think about this.
This is free-form prose.
@UPSERT kailash Kailash'''
    turn = _parse_verb_output(out)
    assert turn.entities == [("kailash", "Kailash")]


def test_parse_accepts_space_after_at():
    """'@ UPSERT priya' still parses (tolerant)."""
    turn = _parse_verb_output("@ UPSERT priya Priya")
    assert turn.entities == [("priya", "Priya")]


def test_parse_bare_at_dropped():
    turn = _parse_verb_output("@\n@UPSERT x X\n@")
    assert turn.entities == [("x", "X")]


def test_parse_english_drift_after_ops_ignored():
    out = '''@UPSERT priya Priya
Wait - that's not correct. Let me reconsider.'''
    turn = _parse_verb_output(out)
    assert turn.entities == [("priya", "Priya")]
    assert turn.statements == []


# --------------------------------------------------------------------
# Non-ingest verbs (edges, retrieval, walks, sys/vault)
# --------------------------------------------------------------------

def test_parse_edge_emits_create_edge():
    """@EDGE produces an entity_edges entry; synthesizer maps each
    slug to the resolver's entity_id before rendering the DSL."""
    turn = _parse_verb_output("@EDGE ent:priya ent:flipkart works_at")
    assert turn.entity_edges == [("priya", "flipkart", "works_at")]
    assert turn.statements == []
    assert turn.entities == []


def test_parse_edge_needs_three_args():
    turn = _parse_verb_output("@EDGE ent:a ent:b")
    assert turn.entity_edges == []
    assert turn.statements == []


def test_parse_remember():
    turn = _parse_verb_output("@REMEMBER what I said about coffee")
    assert turn.statements == ['REMEMBER "what I said about coffee" LIMIT 10']


def test_parse_similar():
    turn = _parse_verb_output("@SIMILAR joining a startup")
    assert turn.statements == ['SIMILAR TO "joining a startup" LIMIT 10']


def test_parse_lexical():
    turn = _parse_verb_output("@LEXICAL python parser bug")
    assert turn.statements == ['LEXICAL SEARCH "python parser bug" LIMIT 10']


def test_parse_answer():
    turn = _parse_verb_output("@ANSWER where does Priya work")
    assert turn.statements == ['ANSWER "where does Priya work"']


def test_parse_recall_walk():
    turn = _parse_verb_output("@RECALL ent:priya")
    assert turn.statements == ['RECALL FROM "ent:priya" DEPTH 2']


def test_parse_traverse_walk():
    turn = _parse_verb_output("@TRAVERSE ent:priya")
    assert turn.statements == ['TRAVERSE FROM "ent:priya" DEPTH 2']


def test_parse_ancestors_walk():
    turn = _parse_verb_output("@ANCESTORS fact:favorite_color")
    assert turn.statements == ['ANCESTORS OF "fact:favorite_color" DEPTH 3']


def test_parse_descendants_walk():
    turn = _parse_verb_output("@DESCENDANTS ent:priya")
    assert turn.statements == ['DESCENDANTS OF "ent:priya" DEPTH 3']


def test_parse_subgraph_walk():
    turn = _parse_verb_output("@SUBGRAPH ent:openai")
    assert turn.statements == ['SUBGRAPH FROM "ent:openai" DEPTH 2']


def test_parse_path_and_shortest():
    assert _parse_verb_output("@PATH ent:a ent:b").statements == [
        'PATH FROM "ent:a" TO "ent:b" MAX_DEPTH 3'
    ]
    assert _parse_verb_output("@SHORTEST_PATH ent:a ent:b").statements == [
        'SHORTEST PATH FROM "ent:a" TO "ent:b"'
    ]
    assert _parse_verb_output("@COMMON ent:a ent:b").statements == [
        'COMMON NEIGHBORS OF "ent:a" AND "ent:b"'
    ]


def test_parse_snapshot_with_name():
    turn = _parse_verb_output('@SNAPSHOT before-cleanup')
    assert turn.statements == ['SYS SNAPSHOT "before-cleanup"']


def test_parse_snapshot_auto_timestamp_when_bare():
    import re
    turn = _parse_verb_output("@SNAPSHOT")
    assert len(turn.statements) == 1
    assert re.fullmatch(
        r'SYS SNAPSHOT "snap-\d{8}T\d{6}Z"',
        turn.statements[0],
    ), turn.statements[0]


def test_parse_rollback_and_snapshots_list():
    assert _parse_verb_output("@ROLLBACK v1").statements == [
        'SYS ROLLBACK TO "v1"'
    ]
    assert _parse_verb_output("@SNAPSHOTS").statements == ['SYS SNAPSHOTS']


def test_parse_compact_optimize():
    assert _parse_verb_output("@COMPACT").statements == ['SYS OPTIMIZE COMPACT']


def test_parse_health_stats_kinds():
    assert _parse_verb_output("@HEALTH").statements == ['SYS HEALTH']
    assert _parse_verb_output("@STATS").statements == ['SYS STATS']
    assert _parse_verb_output("@KINDS").statements == ['SYS KINDS']


def test_parse_explain():
    turn = _parse_verb_output("@EXPLAIN what I said about coffee")
    assert turn.statements == ['SYS EXPLAIN REMEMBER "what I said about coffee"']


def test_parse_mixed_ingest_and_query():
    out = '''@UPSERT priya Priya
@UPSERT openai OpenAI
@REMEMBER what I said about coffee'''
    turn = _parse_verb_output(out)
    assert turn.entities == [("priya", "Priya"), ("openai", "OpenAI")]
    assert turn.statements == ['REMEMBER "what I said about coffee" LIMIT 10']


def test_parse_escapes_quotes_in_query_text():
    turn = _parse_verb_output('@REMEMBER she said "go"')
    assert turn.statements == ['REMEMBER "she said \\"go\\"" LIMIT 10']


def test_parse_query_verb_without_body_dropped():
    turn = _parse_verb_output("@REMEMBER   \n@SIMILAR")
    assert turn.statements == []


def test_parse_walk_verb_without_anchor_dropped():
    turn = _parse_verb_output("@RECALL\n@TRAVERSE")
    assert turn.statements == []


def test_parse_plain_verb_ignores_trailing_tokens():
    """@HEALTH foo still fires; plain handler ignores the rest of the line."""
    turn = _parse_verb_output("@HEALTH ignored")
    assert turn.statements == ['SYS HEALTH']


def test_parse_edge_escapes_quotes_in_ids():
    """Edge slugs + kind survive odd characters; synthesizer escapes
    when it renders the final DSL."""
    turn = _parse_verb_output('@EDGE ent:a ent:b weird"kind')
    assert turn.entity_edges == [("a", "b", 'weird"kind')]


# --------------------------------------------------------------------
# Node lifecycle verbs (update/delete/forget/merge/counterfactual)
# --------------------------------------------------------------------

def test_parse_update_node():
    turn = _parse_verb_output("@UPDATE_NODE me title senior engineer")
    assert turn.statements == [
        'UPDATE NODE "ent:me" SET title = "senior engineer"'
    ]


def test_parse_delete_node():
    turn = _parse_verb_output("@DELETE_NODE obsolete")
    assert turn.statements == ['DELETE NODE "ent:obsolete"']


def test_parse_forget_node():
    turn = _parse_verb_output("@FORGET old_gym")
    assert turn.statements == ['FORGET NODE "ent:old_gym"']


def test_parse_merge_nodes():
    turn = _parse_verb_output("@MERGE maria marie")
    assert turn.statements == [
        'MERGE NODE "ent:maria" INTO "ent:marie"'
    ]


def test_parse_counterfactual():
    turn = _parse_verb_output("@COUNTERFACTUAL joined_stripe")
    assert turn.statements == ['WHAT IF RETRACT "fact:joined_stripe"']


def test_parse_count_nodes_and_edges():
    assert _parse_verb_output("@COUNT_NODES").statements == ['COUNT NODES']
    assert _parse_verb_output("@COUNT_EDGES").statements == ['COUNT EDGES']


# --------------------------------------------------------------------
# DSL synthesis (v5 pre-rendered statements)
# --------------------------------------------------------------------

def test_synthesize_appends_statements_verbatim():
    turn = ParsedTurn(
        entities=[("x", "X")],
        statements=['REMEMBER "hello" LIMIT 3', 'SYS STATS'],
    )
    dsl = _synthesize_dsl(turn, msg_id="m:0", session_id="s", role="user", text="hi")
    assert dsl[-2] == 'REMEMBER "hello" LIMIT 3'
    assert dsl[-1] == 'SYS STATS'


def test_synthesize_statements_only_still_includes_create_node():
    turn = ParsedTurn(statements=['REMEMBER "x" LIMIT 10'])
    dsl = _synthesize_dsl(turn, msg_id="m:0", session_id="s", role="user", text="x")
    assert any(d.startswith('CREATE NODE "m:0"') for d in dsl)
    assert 'REMEMBER "x" LIMIT 10' in dsl


def test_parsed_turn_default_statements_empty():
    turn = ParsedTurn()
    assert turn.statements == []


def test_dsl_escape_handles_quote_and_backslash():
    assert _dsl_escape('he said "hi"') == 'he said \\"hi\\"'
    assert _dsl_escape('c:\\path\\file') == 'c:\\\\path\\\\file'


def test_synthesize_minimal_turn_emits_only_message_node():
    turn = ParsedTurn()
    dsl = _synthesize_dsl(turn, msg_id="m:s1:0", session_id="s1", role="user", text="hi")
    assert len(dsl) == 1
    assert 'CREATE NODE "m:s1:0"' in dsl[0]
    assert 'DOCUMENT "hi"' in dsl[0]


def test_synthesize_with_entities_emits_mention_entity_and_refers_to():
    """Each entity slug yields: mention node + entity node (new) +
    refers_to edge (mention->entity) + mentions edge (msg->mention).
    With no gs passed, every mention mints a fresh entity."""
    turn = ParsedTurn(entities=[("priya", "Priya"), ("openai", "OpenAI")])
    dsl = _synthesize_dsl(turn, msg_id="m:s1:0", session_id="s1", role="user", text="x")
    # 1 message + 2 * (1 mention + 1 entity + 1 refers_to + 1 mentions) = 9
    assert len(dsl) == 9
    text = "\n".join(dsl)
    assert 'kind = "mention"' in text
    assert 'kind = "entity"' in text
    assert 'kind = "refers_to"' in text
    assert 'kind = "mentions"' in text
    assert 'mention:m:s1:0:priya:0' in text
    assert 'mention:m:s1:0:openai:1' in text
    assert 'canonical_name = "Priya"' in text
    assert 'canonical_name = "OpenAI"' in text


def test_synthesize_dedupes_duplicate_entities():
    """Same slug emitted twice in one turn yields exactly one mention
    + one entity + one refers_to."""
    turn = ParsedTurn(entities=[("x", "X"), ("x", "X")])
    dsl = _synthesize_dsl(turn, msg_id="m:0", session_id="s", role="user", text="x")
    mentions = [d for d in dsl if 'kind = "mention"' in d]
    entities = [d for d in dsl if 'kind = "entity"' in d]
    refers = [d for d in dsl if 'kind = "refers_to"' in d]
    assert len(mentions) == 1
    assert len(entities) == 1
    assert len(refers) == 1


def test_synthesize_emits_entity_edges_after_resolution():
    """@EDGE between two upserted slugs renders as entity-to-entity
    after the slug map is populated."""
    turn = ParsedTurn(
        entities=[("priya", "Priya"), ("flipkart", "Flipkart")],
        entity_edges=[("priya", "flipkart", "works_at")],
    )
    dsl = _synthesize_dsl(turn, msg_id="m:0", session_id="s", role="user", text="x")
    edge_lines = [d for d in dsl
                  if d.startswith("CREATE EDGE")
                  and 'kind = "works_at"' in d]
    assert len(edge_lines) == 1
    # Both endpoints must be entity:* ids, not slug literals.
    assert 'entity:' in edge_lines[0]
    assert 'ent:priya' not in edge_lines[0]
    assert 'ent:flipkart' not in edge_lines[0]


def test_synthesize_drops_entity_edge_with_unknown_slug():
    """@EDGE references a slug not declared via @UPSERT; synthesizer
    drops it (logs warning) rather than emitting a broken DSL line."""
    turn = ParsedTurn(
        entities=[("alice", "Alice")],
        entity_edges=[("alice", "ghost", "knows")],
    )
    dsl = _synthesize_dsl(turn, msg_id="m:0", session_id="s", role="user", text="x")
    edge_lines = [d for d in dsl
                  if d.startswith("CREATE EDGE")
                  and 'kind = "knows"' in d]
    assert edge_lines == []


def test_synthesize_belief_and_retract_use_same_fact_id():
    turn = ParsedTurn(
        beliefs=[("fact:drink", "tea")],
        retracts=["fact:drink"],
    )
    dsl = _synthesize_dsl(turn, msg_id="m:1", session_id="s", role="user", text="t")
    retract = next(d for d in dsl if d.startswith("RETRACT"))
    assert '"fact:drink"' in retract
    assert 'superseded by m:1' in retract
    assert any('ASSERT "fact:drink"' in d and 'value = "tea"' in d for d in dsl)


def test_synthesize_escapes_quotes_in_text_and_name():
    turn = ParsedTurn(entities=[("a", 'Alice "Ace"')])
    dsl = _synthesize_dsl(
        turn, msg_id="m:0", session_id="s", role="user",
        text='She said "go".',
    )
    text = "\n".join(dsl)
    assert '\\"go\\"' in text
    assert 'Alice \\"Ace\\"' in text


def test_synthesize_all_together_contract():
    """End-to-end: message + mention/entity/refers_to + belief + retract."""
    turn = ParsedTurn(
        entities=[("priya", "Priya")],
        beliefs=[("fact:color", "green")],
        retracts=["fact:color"],
    )
    dsl = _synthesize_dsl(
        turn, msg_id="m:0", session_id="s1", role="user", text="text",
    )
    text = "\n".join(dsl)
    assert 'kind = "message"' in text
    assert 'kind = "mention"' in text
    assert 'kind = "entity"' in text
    assert 'kind = "refers_to"' in text
    assert 'kind = "mentions"' in text
    assert 'RETRACT "fact:color"' in text
    assert 'ASSERT "fact:color"' in text


def test_ingest_requires_msg_id(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("prompt body")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"")

    ing = BonsaiIngestor(model_path=model, skill_path=skill)
    with pytest.raises(ValueError, match="ingest requires an explicit msg_id"):
        ing.ingest("hello", dry_run=True)


def test_default_prompt_path_ships_in_package(tmp_path: Path):
    """Default prompt file lives inside the package and contains at least one @-verb."""
    from supergraph.bonsai_ingestor import _DEFAULT_PROMPT_PATH
    assert _DEFAULT_PROMPT_PATH.exists()
    body = _DEFAULT_PROMPT_PATH.read_text()
    assert "@UPSERT" in body and "@REMEMBER" in body


# --------------------------------------------------------------------
# Persistent KV cache
# --------------------------------------------------------------------

def test_save_kv_cache_noop_without_path(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("any")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"")

    ing = BonsaiIngestor(model_path=model, skill_path=skill)
    # Should silently no-op when no kv_cache_path configured and no Llama
    ing.save_kv_cache()
    # no crash = pass


def test_save_kv_cache_noop_without_llm(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("any")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"")
    kv = tmp_path / "kv.bin"

    ing = BonsaiIngestor(model_path=model, skill_path=skill, kv_cache_path=kv)
    ing.save_kv_cache()
    assert not kv.exists()


def test_try_load_kv_cache_returns_false_when_missing(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("any")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"x")
    kv = tmp_path / "kv.bin"

    ing = BonsaiIngestor(model_path=model, skill_path=skill, kv_cache_path=kv)
    # Don't need a real Llama - load returns False on missing file before
    # reaching the load_state call.
    assert ing._try_load_kv_cache(None) is False


def test_kv_meta_captures_skill_fingerprint(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("v1 content")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"abcdef")

    ing = BonsaiIngestor(model_path=model, skill_path=skill)
    meta = ing._kv_meta()
    assert meta["skill_fingerprint"] == ing.skill_fingerprint
    assert meta["n_ctx"] == 2048
    assert meta["model_size_bytes"] == 6


def test_try_load_kv_cache_rejects_stale_fingerprint(tmp_path: Path):
    import pickle

    skill = tmp_path / "skill.md"
    skill.write_text("v1")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"x")
    kv = tmp_path / "kv.bin"

    ing = BonsaiIngestor(model_path=model, skill_path=skill, kv_cache_path=kv)
    stale = {
        "meta": {
            "model_path": str(model),
            "model_size_bytes": 1,
            "skill_fingerprint": "deadbeef0000",
            "n_ctx": 2048,
            "chat_format": "qwen",
        },
        "state": "not-a-real-llama-state",
    }
    kv.write_bytes(pickle.dumps(stale))

    # Even with a None Llama, stale meta is detected before load_state is
    # attempted so we return False cleanly.
    assert ing._try_load_kv_cache(None) is False


def test_try_load_kv_cache_handles_corrupt_pickle(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("v1")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"x")
    kv = tmp_path / "kv.bin"
    kv.write_bytes(b"not a pickle at all")

    ing = BonsaiIngestor(model_path=model, skill_path=skill, kv_cache_path=kv)
    assert ing._try_load_kv_cache(None) is False


def test_try_load_kv_cache_handles_wrong_shape(tmp_path: Path):
    import pickle

    skill = tmp_path / "skill.md"
    skill.write_text("v1")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"x")
    kv = tmp_path / "kv.bin"
    kv.write_bytes(pickle.dumps("just a string"))

    ing = BonsaiIngestor(model_path=model, skill_path=skill, kv_cache_path=kv)
    assert ing._try_load_kv_cache(None) is False


# --------------------------------------------------------------------
# NER hints feed Bonsai
# --------------------------------------------------------------------


def _make_ingestor(tmp_path: Path, **kwargs) -> BonsaiIngestor:
    """Build a BonsaiIngestor without touching llama.cpp.

    Skill + model files exist on disk; the LLM is never loaded because we
    only exercise the NER pre-pass and user-message composition.
    """
    skill = tmp_path / "skill.md"
    skill.write_text("v1")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"x")
    return BonsaiIngestor(model_path=model, skill_path=skill, **kwargs)


def test_ner_hints_disabled_when_no_model_dir(tmp_path: Path):
    ing = _make_ingestor(tmp_path)
    assert ing._ner_hints("Maria pushed code with Kailash") == ""


def test_ner_hints_disabled_when_max_hints_zero(tmp_path: Path):
    ing = _make_ingestor(tmp_path,
                         ner_model_dir=tmp_path / "fake-ner",
                         ner_max_hints=0)
    assert ing._ner_hints("Maria pushed code") == ""


def test_ner_hints_formats_unique_entities_in_order(tmp_path: Path, monkeypatch):
    from supergraph.ingest import entity_extract

    class FakeEnt:
        def __init__(self, text, label="PER", score=0.9):
            self.text = text
            self.label = label
            self.score = score

    def fake_extract(text, model_dir=None, score_threshold=0.7, max_length=256):
        return [FakeEnt("Maria"), FakeEnt("OpenAI", "ORG"),
                FakeEnt("maria"), FakeEnt("Berlin", "LOC")]

    monkeypatch.setattr(entity_extract, "extract_entities", fake_extract)

    ing = _make_ingestor(tmp_path, ner_model_dir=tmp_path / "fake-ner")
    out = ing._ner_hints("Maria joined OpenAI in Berlin; Maria was happy.")
    assert out == "[ner:Maria,OpenAI,Berlin]"


def test_ner_hints_caps_at_max_hints(tmp_path: Path, monkeypatch):
    from supergraph.ingest import entity_extract

    class FakeEnt:
        def __init__(self, text):
            self.text = text
            self.label = "PER"
            self.score = 0.9

    monkeypatch.setattr(
        entity_extract, "extract_entities",
        lambda text, model_dir=None, score_threshold=0.7, max_length=256:
            [FakeEnt(c) for c in "ABCDEFGH"],
    )

    ing = _make_ingestor(tmp_path,
                         ner_model_dir=tmp_path / "fake-ner",
                         ner_max_hints=3)
    assert ing._ner_hints("anything") == "[ner:A,B,C]"


def test_ner_hints_empty_results_yield_no_line(tmp_path: Path, monkeypatch):
    from supergraph.ingest import entity_extract

    monkeypatch.setattr(
        entity_extract, "extract_entities",
        lambda text, model_dir=None, score_threshold=0.7, max_length=256: [],
    )

    ing = _make_ingestor(tmp_path, ner_model_dir=tmp_path / "fake-ner")
    assert ing._ner_hints("nothing nameable here") == ""


def test_ner_hints_disable_after_extractor_error(tmp_path: Path, monkeypatch):
    from supergraph.ingest import entity_extract

    calls = {"n": 0}

    def boom(*a, **kw):
        calls["n"] += 1
        raise RuntimeError("ONNX session crashed")

    monkeypatch.setattr(entity_extract, "extract_entities", boom)

    ing = _make_ingestor(tmp_path, ner_model_dir=tmp_path / "fake-ner")
    assert ing._ner_hints("Maria pushed code") == ""
    # Subsequent calls short-circuit without re-invoking the extractor.
    assert ing._ner_hints("Kailash joined OpenAI") == ""
    assert calls["n"] == 1
    assert ing._ner_disabled is True


def test_ner_hints_skip_blank_text_entries(tmp_path: Path, monkeypatch):
    from supergraph.ingest import entity_extract

    class FakeEnt:
        def __init__(self, text):
            self.text = text
            self.label = "PER"
            self.score = 0.9

    monkeypatch.setattr(
        entity_extract, "extract_entities",
        lambda text, model_dir=None, score_threshold=0.7, max_length=256:
            [FakeEnt(""), FakeEnt("   "), FakeEnt("Alice")],
    )

    ing = _make_ingestor(tmp_path, ner_model_dir=tmp_path / "fake-ner")
    assert ing._ner_hints("anything") == "[ner:Alice]"
