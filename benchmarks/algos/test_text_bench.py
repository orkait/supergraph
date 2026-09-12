"""Text primitives micro-benchmarks."""

from supergraph.algos.text import fts5_sanitize, tokenize_unicode


class TestFts5Sanitize:
    def test_short(self, benchmark):
        benchmark(fts5_sanitize, "what is the user's favorite food?")

    def test_long(self, benchmark):
        q = "find all references to Kubernetes and Docker in recent messages " * 20
        benchmark(fts5_sanitize, q)

    def test_reserved_heavy(self, benchmark):
        benchmark(fts5_sanitize, "AND OR NOT NEAR special keywords mixed AND more")

    def test_batch(self, benchmark, fts5_queries):
        def _batch():
            return [fts5_sanitize(q) for q in fts5_queries]
        benchmark(_batch)


class TestTokenizeUnicode:
    def test_short(self, benchmark):
        benchmark(tokenize_unicode, "the quick brown fox jumps over the lazy dog")

    def test_long(self, benchmark):
        q = "the quick brown fox jumps over the lazy dog " * 100
        benchmark(tokenize_unicode, q)
