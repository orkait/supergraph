"""q.raw() escape hatch + plugin registry."""
import pytest

from supergraph import q, register_verb, Query


class TestRaw:
    def test_plain_passthrough(self):
        assert q.raw("NODES").dsl() == "NODES"

    def test_param_substitution(self):
        out = q.raw('CREATE NODE :id kind = :k', id="mem:1", k="memory").dsl()
        assert out == 'CREATE NODE "mem:1" kind = "memory"'

    def test_param_escape_injection(self):
        out = q.raw('CREATE NODE :id kind = :k DOCUMENT :doc',
                    id="mem:1", k="memory", doc='foo"; DROP; --').dsl()
        assert r'\"' in out

    def test_missing_param_raises(self):
        with pytest.raises(ValueError, match="missing params"):
            q.raw("CREATE NODE :id", )  # no kwargs

    def test_extra_param_raises(self):
        with pytest.raises(ValueError, match="unused params"):
            q.raw("NODES", extra="x")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            q.raw("")


class TestPluginRegistry:
    def test_register_and_call(self):
        @register_verb("my_custom_test_verb")
        def my_custom_test_verb(id: str, *, level: int) -> Query:
            return Query(
                _verb="raw",
                _params={"text": f'MY CUSTOM VERB "{id}" LEVEL {level}'},
                _kind="read",
            )

        out = q.my_custom_test_verb("n1", level=3)
        assert out.dsl() == 'MY CUSTOM VERB "n1" LEVEL 3'

    def test_unknown_verb_raises(self):
        with pytest.raises(AttributeError, match="has no verb"):
            q.definitely_not_a_real_verb

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError):
            register_verb("not valid ident")(lambda: None)
