import json

from superclaw.reasoning import detect_reasoning, request_extras, resolve_reasoning

BONSAI = ("low", "medium", "xhigh")


def test_maps_abstract_effort_to_the_nearest_accepted_level():
    assert resolve_reasoning("high", BONSAI, True) == {"reasoning_effort": "xhigh"}
    assert resolve_reasoning("medium", BONSAI, True) == {"reasoning_effort": "medium"}
    assert resolve_reasoning("low", BONSAI, True) == {"reasoning_effort": "low"}


def test_off_disables_thinking_when_the_model_supports_it():
    assert resolve_reasoning("off", BONSAI, True) == {"chat_template_kwargs": {"enable_thinking": False}}
    assert resolve_reasoning("", BONSAI, True) == {"chat_template_kwargs": {"enable_thinking": False}}


def test_off_falls_to_the_lowest_level_when_thinking_cannot_be_disabled():
    assert resolve_reasoning("off", ("low", "medium", "high"), False) == {"reasoning_effort": "low"}


def test_unknown_model_passes_the_effort_through_untranslated():
    assert resolve_reasoning("high", (), False) == {"reasoning_effort": "high"}
    assert resolve_reasoning("off", (), False) == {}


def test_accepted_level_is_used_verbatim_when_the_model_lists_it():
    assert resolve_reasoning("high", ("low", "medium", "high", "xhigh"), True) == {"reasoning_effort": "high"}


def test_detect_reads_accepted_set_and_thinking_switch_from_the_template():
    tmpl = ("{%- set r = reasoning_effort|default('xhigh') %}"
            "{%- if r not in ('xhigh', 'medium', 'low') %}{{ raise }}{%- endif %}"
            "{%- if enable_thinking is false %}{%- endif %}")
    fetch = lambda url: json.dumps({"chat_template": tmpl}).encode()
    accepted, can_disable = detect_reasoning("http://127.0.0.1:8081/v1/", fetch)
    assert set(accepted) == {"low", "medium", "xhigh"} and can_disable is True


def test_detect_returns_unknown_when_props_is_absent_or_not_a_reasoning_model():
    boom = lambda url: (_ for _ in ()).throw(OSError("404"))
    assert detect_reasoning("http://x/v1", boom) == ((), False)
    plain = lambda url: json.dumps({"chat_template": "{{ messages }}"}).encode()
    assert detect_reasoning("http://x/v1", plain) == ((), False)


def test_detect_derives_the_props_url_from_the_base():
    seen = []

    def fetch(url):
        seen.append(url)
        return json.dumps({"chat_template": "reasoning_effort not in ('low', 'high')"}).encode()

    detect_reasoning("http://127.0.0.1:8081/v1", fetch)
    assert seen == ["http://127.0.0.1:8081/props"]


def test_request_extras_caches_detection_per_base_url():
    calls = []

    def fetch(url):
        calls.append(url)
        return json.dumps({"chat_template": "reasoning_effort not in ('low','medium','xhigh'); enable_thinking"}).encode()

    cache: dict = {}
    a = request_extras("high", "http://h/v1", fetch, cache)
    b = request_extras("off", "http://h/v1", fetch, cache)
    assert a == {"reasoning_effort": "xhigh"} and b == {"chat_template_kwargs": {"enable_thinking": False}}
    assert len(calls) == 1


def test_request_extras_passes_through_when_there_is_no_base_url():
    assert request_extras("high", None, None, {}) == {"reasoning_effort": "high"}
    assert request_extras("off", None, None, {}) == {}
    assert request_extras("", None, None, {}) == {}
