import json
from types import SimpleNamespace

from superclaw.loop import Options, run
from superclaw.models import ModelInfo, lookup
from superclaw.policy import Mode, Policy
from superclaw.provider import parse_response
from superclaw.runtime import Completion, ToolCall, Usage, approx_tokens
from superclaw.settings import LIMITS, Settings
from superclaw.tools import Registry
from superclaw.tools.files import core_file_tools


def test_approx_tokens_charges_ink_quarters_and_foreign_bytes():
    assert approx_tokens("abcd efgh") == 2
    assert approx_tokens("日本") == 6
    assert approx_tokens("   ") == 0


def test_lookup_resolves_through_provider_prefixes_and_falls_back():
    info = lookup("openrouter/deepseek/deepseek-v4-flash")
    assert info.known and info.context_window > LIMITS.context_window_fallback and info.input_per_token > 0
    assert lookup("bogus-gateway/deepseek-v4-flash").known
    unknown = lookup("nobody/no-such-model")
    assert not unknown.known and unknown.context_window == LIMITS.context_window_fallback


def test_cost_prices_cached_input_at_the_cache_rate():
    info = ModelInfo("m", 1000, 100, input_per_token=1.0, output_per_token=10.0, cache_read_per_token=0.1)
    assert info.cost(Usage(input_tokens=100, output_tokens=1, cache_read_tokens=40)) == 60 + 4 + 10


def test_settings_window_prefers_override_then_catalog():
    env = {"SUPERCLAW_MODEL": "openrouter/deepseek/deepseek-v4-flash", "SUPERCLAW_BUDGET_USD": "0.5"}
    s = Settings.from_env(env)
    assert s.context_window == 0 and s.window() == lookup(s.model).context_window and s.budget_usd == 0.5
    assert Settings.from_env({**env, "SUPERCLAW_CONTEXT_WINDOW": "4096"}).window() == 4096


def test_parse_response_reads_cached_prompt_tokens():
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=5, prompt_tokens_details=SimpleNamespace(cached_tokens=30)),
    )
    assert parse_response(resp).usage.cache_read_tokens == 30


def test_usd_budget_stops_the_run_and_reports_cost(tmp_path):
    class Scripted:
        def complete(self, messages, tools):
            return Completion(tool_calls=[ToolCall("c", "read_file", json.dumps({"path": "nope"}))], usage=Usage(1000, 100))

    reg = Registry()
    for t in core_file_tools():
        reg.register(t)
    events = []
    info = ModelInfo("m", 100_000, 4096, input_per_token=0.001, output_per_token=0.002)
    res = run("go", Scripted(), Options(registry=reg, policy=Policy(tmp_path, Mode.AUTO), workspace=tmp_path, system_prompt="S",
                                        budget_usd=2.0, model_info=info, context_window=100_000, on_event=events.append))
    usage = [e for e in events if e["type"] == "usage"]
    assert usage[0]["cost_usd"] == 1.2 and usage[0]["context_window"] == 100_000
    assert res.stop_reason == "budget" and "$2.00" in res.final_answer and res.turns == 3
