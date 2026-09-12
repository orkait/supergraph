import pytest

from superclaw.compaction import RESUME_NOTE, SUMMARY_LABEL, compact, project, render_transcript, threshold
from superclaw.guards import Guards, ends_with_continuation_cue, error_signature
from superclaw.runtime import Message, ToolCall, approx_tokens, estimate_tokens, to_wire


def user(text):
    return Message(role="user", content=text)


def assistant(text="", calls=()):
    return Message(role="assistant", content=text, tool_calls=list(calls))


def tool(call_id, text, is_error=False):
    return Message(role="tool", content=text, tool_call_id=call_id, is_error=is_error)


class TestRuntime:
    def test_to_wire_shapes(self):
        msgs = [
            Message(role="system", content="sys"),
            user("hi"),
            assistant("", [ToolCall("c1", "read_file", '{"path": "a"}')]),
            tool("c1", "1→x"),
            assistant("done"),
        ]
        assert to_wire(msgs) == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "1→x"},
            {"role": "assistant", "content": "done"},
        ]

    def test_approx_tokens_counts_non_whitespace_quarters(self):
        assert approx_tokens("") == 0
        assert approx_tokens("abcd efgh") == 2
        assert approx_tokens("a" * 400) == 100

    def test_estimate_tokens_includes_tools_and_calls(self):
        msgs = [user("a" * 40), assistant("", [ToolCall("c", "t", "x" * 40)])]
        tools = [{"type": "function", "function": {"name": "t", "description": "d" * 40, "parameters": {}}}]
        assert estimate_tokens(msgs, tools) > estimate_tokens(msgs, [])
        assert estimate_tokens(msgs, []) >= 20


class TestCompaction:
    def test_threshold_is_seventy_percent(self):
        assert threshold(1000) == 700
        assert threshold(0) == 0

    def test_short_history_is_untouched(self):
        msgs = [Message(role="system", content="s"), user("a"), assistant("b")]
        res = compact(msgs, preserve_last=6, summarize=lambda m: "SUM")
        assert not res.compacted
        assert res.messages == msgs

    def test_summarizes_middle_keeps_system_and_suffix(self):
        msgs = [Message(role="system", content="s")]
        for i in range(6):
            msgs += [user(f"u{i}"), assistant(f"a{i}")]
        seen = []

        def summarize(middle):
            seen.append(middle)
            return "SUM"

        res = compact(msgs, preserve_last=3, summarize=summarize)
        assert res.compacted
        assert res.messages[0].content == "s"
        assert res.messages[1].role == "user"
        assert res.messages[1].content.startswith(SUMMARY_LABEL + "\nSUM")
        assert res.messages[1].content.rstrip().endswith(RESUME_NOTE)
        assert res.messages[2].role == "assistant"
        assert [m.content for m in res.messages[2:]] == ["a4", "u5", "a5"]
        assert all(f"[user #{i}]\nu{i // 2}" in seen[0] for i in range(0, 9, 2))
        assert res.removed == 9

    def test_suffix_never_starts_on_tool_result(self):
        msgs = [Message(role="system", content="s"), user("u0"), assistant("a0"), user("u1"),
                assistant("", [ToolCall("c", "t", "{}")]), tool("c", "r"), assistant("a1")]
        res = compact(msgs, preserve_last=2, summarize=lambda m: "SUM")
        assert res.messages[2].role == "assistant"
        assert res.messages[2].tool_calls[0].id == "c"

    def test_preserved_state_and_projection(self):
        msgs = [Message(role="system", content="s"), user(f"{SUMMARY_LABEL}\nOLD FACTS"),
                assistant("", [ToolCall("c1", "skill", '{"name": "bench"}'), ToolCall("c2", "read_file", '{"path": "a"}')]),
                tool("c1", "skill body"), tool("c2", "file text"),
                assistant("", [ToolCall("c3", "edit_file", '{"path": "src/x.py", "old_string": "a", "new_string": "b"}')]),
                tool("c3", "Error: old_string not found", is_error=True),
                user("u"), assistant("a"), user("u2"), assistant("a2")]
        brief = project(msgs[1:7])
        assert brief.startswith("[previous summary]\nOLD FACTS")
        assert "skill body" not in brief and "file text" not in brief
        assert "[tool_error #5] edit_file\nError: old_string not found" in brief
        res = compact(msgs, preserve_last=2, summarize=lambda b: "SUM", plan_text="Current Plan:\n1. [pending] x")
        body = res.messages[1].content
        assert "Current Plan:" in body and "Skills loaded: bench" in body and "Files edited: src/x.py" in body

    def test_render_transcript_clamps_tool_output(self):
        text = render_transcript([assistant("", [ToolCall("c", "t", "a" * 1000)]), tool("c", "b" * 5000)])
        assert "a" * 500 in text and "a" * 600 not in text
        assert "b" * 2000 in text and "b" * 2100 not in text


class TestGuards:
    @pytest.mark.parametrize("text,expected", [
        ("Now let me check the config:", True),
        ("Done. Next, I'll run the tests:", True),
        ("Here is the summary:", False),
        ("Let me know if you need anything", False),
        ("All good.", False),
        ("", False),
    ])
    def test_continuation_cue(self, text, expected):
        assert ends_with_continuation_cue(text) is expected

    def test_empty_turns_stop_after_three(self):
        g = Guards()
        assert g.observe_turn("", 0) is False
        assert g.observe_turn("", 0) is False
        assert g.observe_turn("", 0) is True

    def test_visible_output_resets_empty_streak(self):
        g = Guards()
        g.observe_turn("", 0)
        g.observe_turn("text", 0)
        g.observe_turn("", 0)
        assert g.observe_turn("", 0) is False

    def test_same_error_streak_hints_then_stops(self):
        g = Guards()
        outcomes = [g.observe_tool_result("edit_file", True, "Error: old_string not found in a.py") for _ in range(6)]
        assert [o.hint for o in outcomes] == [False, True, False, False, False, False]
        assert [o.stop for o in outcomes] == [False, False, False, False, False, True]
        assert outcomes[-1].count == 6

    def test_streak_resets_on_success_or_different_error(self):
        g = Guards()
        g.observe_tool_result("bash", True, "Error: x")
        g.observe_tool_result("bash", True, "Error: y")
        assert g.observe_tool_result("bash", True, "Error: y").count == 2
        g.observe_tool_result("bash", False, "ok")
        assert g.observe_tool_result("bash", True, "Error: y").count == 1

    def test_error_signature_ignores_numbers_and_paths(self):
        assert error_signature("Error: line 12 in /tmp/a/x.py") == error_signature("Error: line 99 in /tmp/b/y.py")

    def test_stale_plan_reminder_fires_once(self):
        g = Guards(stale_tool_calls=3)
        g.observe_tool_call("update_plan")
        for _ in range(3):
            g.observe_tool_call("read_file")
        assert g.stale_plan_reminder(pending=True) is not None
        assert g.stale_plan_reminder(pending=True) is None
        g.observe_tool_call("update_plan")
        for _ in range(3):
            g.observe_tool_call("read_file")
        assert g.stale_plan_reminder(pending=True) is not None

    def test_stale_plan_reminder_needs_pending_items(self):
        g = Guards(stale_tool_calls=1)
        g.observe_tool_call("read_file")
        assert g.stale_plan_reminder(pending=False) is None

    def test_tool_only_reminder_after_six_silent_turns(self):
        g = Guards()
        for _ in range(5):
            assert g.tool_only_reminder("", 2) is None
        assert g.tool_only_reminder("", 2) is not None
        assert g.tool_only_reminder("", 2) is None
