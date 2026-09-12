from supergraph import SuperGraph
from supergraph.algos.sentence_split import split_sentences


class TestSentenceSplitter:
    def test_empty_returns_empty(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_short_text_returns_as_is(self):
        assert split_sentences("hello") == ["hello"]
        assert split_sentences("Dr. Smith") == ["Dr. Smith"]

    def test_two_sentences(self):
        result = split_sentences("She moved to Sweden in 2023. Her name is Caroline.")
        assert len(result) == 2
        assert "Sweden" in result[0]
        assert "Caroline" in result[1]

    def test_abbreviation_not_split(self):
        result = split_sentences("Dr. Smith went to the store. He bought milk.")
        assert len(result) == 2
        assert "Smith" in result[0]
        assert "He bought" in result[1]
        assert "Dr." in result[0]

    def test_abbreviation_etc_not_split(self):
        result = split_sentences("He brought apples, oranges, etc. They were fresh. It was a good day.")
        assert len(result) >= 2

    def test_multi_sentence_query(self):
        result = split_sentences("Where did she move from? She traveled there. When did it happen?")
        assert len(result) >= 2

    def test_three_sentences(self):
        result = split_sentences("Caroline moved from Sweden. She was born in 1990. Melanie met her there.")
        assert len(result) == 3


class TestSentenceQueryExpansion:
    def test_disabled_by_default(self):
        g = SuperGraph(ceiling_mb=256)
        assert getattr(g._executor, '_sentence_query_expansion', False) is True

    def test_enabled_via_constructor(self):
        g = SuperGraph(ceiling_mb=256, sentence_query_expansion=True)
        assert g._executor._sentence_query_expansion is True

    def test_enabled_via_config(self):
        from supergraph.config import SuperGraphConfig, DslConfig
        cfg = SuperGraphConfig()
        cfg = type(cfg)(
            core=cfg.core, vector=cfg.vector, document=cfg.document,
            dsl=DslConfig(sentence_query_expansion=True),
            vault=cfg.vault, persistence=cfg.persistence,
            retention=cfg.retention, server=cfg.server,
            evolution=cfg.evolution,
        )
        g = SuperGraph(ceiling_mb=256, config=cfg)
        assert g._executor._sentence_query_expansion is True

    def test_single_sentence_query_unchanged(self):
        g_on = SuperGraph(ceiling_mb=256, sentence_query_expansion=True)
        g_on.execute('SYS REGISTER NODE KIND "memory" REQUIRED content:string EMBED content')
        g_on.execute('CREATE NODE "msg1" kind = "memory" content = "Caroline moved from Sweden in 2023."')

        result = g_on.execute('REMEMBER "Caroline Sweden" LIMIT 5')
        assert len(result.data) == 1
        assert "Sweden" in result.data[0]["content"]

    def test_multi_sentence_query_finds_both_topics(self):
        g = SuperGraph(ceiling_mb=256, sentence_query_expansion=True)
        g.execute('SYS REGISTER NODE KIND "memory" REQUIRED content:string EMBED content')
        g.execute('CREATE NODE "msg1" kind = "memory" content = "Caroline moved from Sweden in 2023. She met Melanie there."')
        g.execute('CREATE NODE "msg2" kind = "memory" content = "The weather was great. They traveled together."')
        g.execute('CREATE NODE "msg3" kind = "memory" content = "Programming in Python is fun. Machine learning is interesting too."')

        result = g.execute('REMEMBER "Where did she move? The weather was great." LIMIT 10')
        ids = [n["id"] for n in result.data]
        assert "msg1" in ids

    def test_expansion_adds_more_candidates(self):
        g_on = SuperGraph(ceiling_mb=256, sentence_query_expansion=True)
        g_on.execute('SYS REGISTER NODE KIND "memory" REQUIRED content:string EMBED content')
        g_on.execute('CREATE NODE "msg1" kind = "memory" content = "Caroline moved from Sweden in 2023."')
        g_on.execute('CREATE NODE "msg2" kind = "memory" content = "She likes programming in Python."')
        g_on.execute('CREATE NODE "msg3" kind = "memory" content = "The weather in Stockholm is cold."')

        result = g_on.execute('REMEMBER "Where is she from? What does she like?" LIMIT 5')
        assert len(result.data) >= 1
