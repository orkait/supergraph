"""Integration tests for pipeline refactoring."""
from supergraph import SuperGraph


class TestSentenceLevelIngest:
    def test_sentence_nodes_created(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('SYS REGISTER NODE KIND "message" REQUIRED content:string EMBED content')
        g.execute('CREATE NODE "msg0" kind = "message" content = "Caroline moved from Sweden. She studied at KTH."')

        # Check sentence child nodes exist
        sentences = g.execute('NODES WHERE kind = "sentence"')
        assert len(sentences.data) >= 2
        g.close()

    def test_message_has_vector_alongside_sentences(self):
        """Message nodes get vectors AND sentence child nodes are created."""
        g = SuperGraph(ceiling_mb=256)
        g.execute('SYS REGISTER NODE KIND "message" REQUIRED content:string EMBED content')
        g.execute('CREATE NODE "msg0" kind = "message" content = "Hello world. Goodbye world."')

        msg = g.execute('NODE "msg0"')
        assert msg.data is not None
        assert g._vector_store is not None
        slot = g._store.id_to_slot[g._store.string_table.intern("msg0")]
        assert g._vector_store.has_vector(slot)
        sentences = g.execute('NODES WHERE kind = "sentence"')
        assert len(sentences.data) >= 2
        g.close()


class TestThreeSignalFusion:
    def test_three_weight_fusion(self):
        g = SuperGraph(ceiling_mb=256, sentence_query_expansion=True)
        g.execute('SYS REGISTER NODE KIND "message" REQUIRED content:string EMBED content')
        g.execute('CREATE NODE "m1" kind = "message" content = "Caroline moved to Sweden."')
        g.execute('CREATE NODE "m2" kind = "message" content = "The weather was great."')

        result = g.execute('REMEMBER "Sweden" LIMIT 5')
        assert len(result.data) >= 1
        assert "_remember_score" in result.data[0]
        assert "_vector_sim" in result.data[0]
        assert "_bm25_score" in result.data[0]
        assert "_recency_score" in result.data[0]
        g.close()


class TestRerankerIntegration:
    def test_reranker_not_configured(self):
        """Without reranker, pipeline returns top-K from fusion."""
        g = SuperGraph(ceiling_mb=256, sentence_query_expansion=True)
        g.execute('SYS REGISTER NODE KIND "message" REQUIRED content:string EMBED content')
        for i in range(5):
            g.execute(f'CREATE NODE "m{i}" kind = "message" content = "Test message {i} about topic {i}."')

        result = g.execute('REMEMBER "topic 0" LIMIT 2')
        assert len(result.data) == 2
        g.close()


class TestNucleusExpansion:
    def test_nucleus_disabled_by_default(self):
        g = SuperGraph(ceiling_mb=256)
        assert g._executor._nucleus_expansion is False
        g.close()

    def test_nucleus_returns_separate_meta(self):
        g = SuperGraph(ceiling_mb=256, nucleus_expansion=True)
        g.execute('SYS REGISTER NODE KIND "message" REQUIRED content:string EMBED content')
        g.execute('CREATE NODE "m1" kind = "message" content = "First message about Caroline."')
        g.execute('CREATE NODE "m2" kind = "message" content = "Second message, she continued."')
        g.execute('CREATE EDGE "m1" -> "m2" kind = "next"')

        result = g.execute('REMEMBER "Caroline" LIMIT 1')
        assert len(result.data) == 1
        # Nucleus should be in meta, not in data
        assert "nucleus" in result.meta
        g.close()
