from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
from collections.abc import Callable

from superclaw.agents import Agent
from superclaw.compaction import SUMMARY_INSTRUCTIONS, compact, cut_point, prune_tool_results
from superclaw.delegate import SPAWN_KEY
from superclaw.hooks import Dispatcher
from superclaw.guards import (
    DROPPED_TOOL_CALL_NOTICE,
    EMPTY_TURN_NUDGE,
    MAX_TURNS_FINAL_ANSWER_PROMPT,
    FailureOutcome,
    Guards,
    calls_per_turn_reminder,
    continue_nudge,
    ends_with_continuation_cue,
    ends_with_promise,
    no_output_stop_answer,
    promise_nudge,
    tool_failure_hint,
    tool_failure_stop_answer,
)
from superclaw.meter import ContextMeter, bounded
from superclaw.models import ModelInfo
from superclaw.policy import Action, Policy, validate_prefix
from superclaw.prompt import agent_block
from superclaw.runtime import Completion, Message, Provider, ToolCall, Usage, approx_tokens, clip, estimate_tokens
from superclaw.session import SessionStore, prompt_hash
from superclaw.settings import LIMITS, TOUCH_VERBS
from superclaw.tools import PathEscapes, Registry, Result as ToolResult, ToolContext, jail
from superclaw.tools.ask import NON_INTERACTIVE_MESSAGE, parse_questions
from superclaw.tools.plan import format_plan, pending_items
from superclaw.verifier import verify

ABORTED_TOOL_RESULT = "Aborted: an earlier tool call halted the run."
OWN_STATE_TOOLS = {"update_plan", "ask_user", "memory_note", "write_file", "edit_file", "submit_spec"}


def label_untrusted(tool: str, output: str) -> str:
    if tool in OWN_STATE_TOOLS or output.startswith("Error:"):
        return output
    return f'<untrusted source="{tool}">\n{output}\n</untrusted>'


@dataclass
class Options:
    registry: Registry
    policy: Policy
    workspace: Path
    extra_dirs: tuple[Path, ...] = ()
    images: list[str] = field(default_factory=list)
    system_prompt: str = ""
    history: list[Message] = field(default_factory=list)
    max_turns: int = LIMITS.max_turns
    token_budget: int = 0
    budget_usd: float = 0.0
    context_window: int = 0
    model_info: ModelInfo | None = None
    reserve_tokens: int = LIMITS.compaction_reserve_tokens
    keep_tokens: int = LIMITS.compaction_keep_tokens
    require_completion_signal: bool = False
    verify: bool = False
    on_event: Callable[[dict[str, Any]], None] | None = None
    on_permission: Callable[[dict[str, Any]], str] | None = None
    on_ask_user: Callable[[list[dict[str, Any]]], list[str]] | None = None
    agents: dict[str, Agent] = field(default_factory=dict)
    session: SessionStore | None = None
    session_id: str = ""
    summarize: Callable[[str], str] | None = None
    hooks: Dispatcher | None = None
    session_start: bool = True
    depth: int = 0
    cancelled: Callable[[], bool] | None = None


@dataclass
class Result:
    final_answer: str
    turns: int
    messages: list[Message]
    incomplete: bool = False
    incomplete_reason: str = ""
    stop_reason: str = ""
    saved_tokens: int = 0
    kept_out_tokens: int = 0


class _Run:
    def __init__(self, provider: Provider, options: Options) -> None:
        self.provider = provider
        self.o = options
        self.guards = Guards()
        self.meter = ContextMeter(options.context_window, options.reserve_tokens)
        self.keep_tokens = bounded(options.keep_tokens, options.context_window)
        self.messages: list[Message] = []
        self.seqs: list[int] = []
        self.turns = 0
        self.tokens_used = 0
        self.cost_usd = 0.0
        self.loaded: set[str] = set()
        self.nudges = 0
        self.promise_nudged = False
        self.objective = ""
        self.changed: set[str] = set()
        self.refs: list[str] = []
        self.saved_tokens = 0
        self.kept_out_tokens = 0
        self.control = ""
        plan = options.session.plan(options.session_id) if options.session and options.session_id else []
        self.ctx = ToolContext(workspace=options.workspace, session_id=options.session_id, extra_dirs=options.extra_dirs,
                               state={"plan": plan, SPAWN_KEY: self.spawn})

    def emit(self, event: dict[str, Any]) -> None:
        if self.o.on_event:
            self.o.on_event(event)

    def persist(self, etype: str, payload: dict[str, Any]) -> int:
        if self.o.session and self.o.session_id:
            return self.o.session.append(self.o.session_id, etype, payload)
        return 0

    def touch(self, path: str, verb: str) -> None:
        if not (self.o.session and self.o.session_id and path):
            return
        try:
            target = jail(self.ctx.roots, path)
        except PathEscapes:
            return
        self.o.session.touch(self.o.session_id, str(target), verb)

    def append(self, message: Message) -> None:
        self.messages.append(message)
        self.meter.append(message)
        self.ctx.files.cursor = len(self.messages)
        if message.role == "tool":
            seq = self.persist("tool_result", {"tool_call_id": message.tool_call_id, "output": message.content, "ok": not message.is_error})
        else:
            payload: dict[str, Any] = {
                "role": message.role, "content": message.content,
                "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in message.tool_calls],
            }
            if message.images:
                payload["images"] = len(message.images)
            seq = self.persist("message", payload)
        self.seqs.append(seq)

    def complete(self, exposed: list[dict[str, Any]]) -> Completion:
        try:
            if getattr(self.provider, "streams", False) and self.o.on_event:
                return self.provider.complete(self.messages, exposed, on_text=lambda text: self.emit({"type": "text_delta", "text": text}))
            return self.provider.complete(self.messages, exposed)
        except Exception as e:
            self.persist("error", {"turn": self.turns, "error": f"{type(e).__name__}: {e}"})
            self.emit({"type": "error", "message": str(e), "recoverable": False})
            raise

    def result(self, answer: str, **kw: Any) -> Result:
        return Result(final_answer=answer, turns=self.turns, messages=list(self.messages), saved_tokens=self.saved_tokens, kept_out_tokens=self.kept_out_tokens, **kw)

    def summarize(self, brief: str) -> str:
        if self.o.summarize:
            return self.o.summarize(brief)
        request = [Message(role="system", content=SUMMARY_INSTRUCTIONS), Message(role="user", content=brief)]
        return self.provider.complete(request, []).text

    def maybe_compact(self, exposed: list[dict[str, Any]]) -> None:
        if not self.meter.pressure(estimate_tokens(self.messages, exposed)):
            return
        if self.prune() and not self.meter.pressure(estimate_tokens(self.messages, exposed)):
            return
        plan = self.ctx.state.get("plan", [])
        res = compact(self.messages, keep_tokens=self.keep_tokens, summarize=self.summarize,
                      plan_text=format_plan(plan) if plan else "")
        if not res.compacted:
            return
        system_end = sum(1 for m in self.messages if m.role == "system")
        through = self.seqs[system_end + res.removed - 1]
        summary_seq = self.persist("compaction", {"summary": res.summary, "through_seq": through})
        self.ctx.files.compacted(system_end, res.removed)
        self.messages = res.messages
        self.seqs = [*self.seqs[:system_end], summary_seq, *self.seqs[system_end + res.removed:]]
        self.meter.reset()
        self.emit({"type": "compaction", "removed": res.removed})

    def prune(self) -> int:
        pruned = prune_tool_results(self.messages, cut_point(self.messages, self.keep_tokens))
        for index, content, _ in pruned:
            self.saved_tokens += approx_tokens(self.messages[index].content) - approx_tokens(content)
            self.messages[index].content = content
            self.persist("prune", {"seq": self.seqs[index], "output": content})
        if pruned:
            self.ctx.files.evict({index for index, _, _ in pruned})
            self.meter.reset()
            self.emit({"type": "prune", "results": len(pruned), "refs": [ref for _, _, ref in pruned if ref]})
        return len(pruned)

    def decide(self, name: str, args: dict[str, Any]) -> tuple[bool, str]:
        tool = self.o.registry.get(name)
        if tool is None:
            return False, f"unknown tool {name!r}"
        decision = self.o.policy.evaluate(tool, args)
        self.ctx.state["approval"] = {"escalated": decision.escalated, "network": decision.network}
        if decision.action == Action.ALLOW:
            return True, decision.reason
        if decision.action == Action.DENY:
            return False, decision.reason
        prefix = args.get("prefix_rule") or []
        prefix_error = validate_prefix(prefix, str(args.get("command") or "")) if prefix else "no prefix_rule offered"
        request = {"tool": name, "args": args, "reason": decision.reason, "risk": decision.risk.level,
                   "categories": decision.risk.categories, "prefix": prefix if not prefix_error else []}
        self.emit({"type": "permission_request", **request})
        if self.o.on_permission is None:
            return False, f"no interactive approver; {decision.reason}"
        choice = self.o.on_permission(request)
        self.emit({"type": "permission_decision", "tool": name, "decision": choice})
        if choice == "allow_session":
            self.o.policy.grant_session(name)
        if choice == "allow_prefix":
            if prefix_error:
                return False, f"prefix not remembered: {prefix_error}"
            self.o.policy.grant_prefix(prefix)
        if choice in ("allow", "allow_session", "allow_prefix"):
            return True, "approved"
        return False, "approval declined"

    def execute(self, call: ToolCall) -> tuple[ToolResult, bool]:
        try:
            args = json.loads(call.arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as e:
            return ToolResult.error(f"Error: invalid JSON arguments for {call.name}: {e}"), False
        self.emit({"type": "tool_call", "id": call.id, "name": call.name, "args": args})
        self.guards.observe_tool_call(call.name)
        if call.name == "ask_user" and self.o.on_ask_user:
            return self.ask_user(args), False
        allowed, reason = self.decide(call.name, args)
        if not allowed:
            return ToolResult.error(f"Error: {call.name} denied: {reason}"), True
        if self.o.hooks:
            before = self.o.hooks.dispatch("beforeTool", {"tool": call.name, "id": call.id, "args": args}, call.name)
            if before.blocked:
                return ToolResult.error(f"Error: {call.name} blocked by hook {before.blocked_by}: {' '.join(before.context)}".rstrip(": ")), True
        res = self.o.registry.run(call.name, args, self.ctx, call.id)
        if call.name == "update_plan" and res.ok:
            self.persist("plan", {"items": self.ctx.state.get("plan", [])})
        if res.ok and call.name in TOUCH_VERBS:
            self.touch(str(args.get("path") or ""), TOUCH_VERBS[call.name])
        if self.o.hooks:
            after = self.o.hooks.dispatch("afterTool", {"tool": call.name, "id": call.id, "args": args, "ok": res.ok, "output": res.output[:LIMITS.hook_output_chars]}, call.name)
            if after.context:
                res.output += "\n\n[hook] " + "\n[hook] ".join(after.context)
                res = self.o.registry.finalize(call.name, args, res, self.ctx, call.id)
        return res, False

    def ask_user(self, args: dict[str, Any]) -> ToolResult:
        try:
            questions = parse_questions(args)
        except ValueError as e:
            return ToolResult.error(f"Error: invalid arguments for ask_user: {e}")
        if self.o.on_ask_user is None:
            return ToolResult.success(NON_INTERACTIVE_MESSAGE)
        answers = self.o.on_ask_user(questions)
        answers += [""] * (len(questions) - len(answers))
        return ToolResult.success("\n".join(f"Q: {q['question']}\nA: {a}" for q, a in zip(questions, answers, strict=True)))

    def incomplete_reason(self, text: str) -> str:
        pending = pending_items(self.ctx.state)
        if pending:
            return f"{len(pending)} plan item(s) still pending"
        if ends_with_continuation_cue(text):
            return "the message ends mid-step"
        return ""

    def nudge(self, text: str, reason: str, instruction: str) -> Result | None:
        if self.nudges < LIMITS.max_continue_nudges:
            self.nudges += 1
            self.append(Message(role="user", content=instruction))
            return None
        return self.result(text, incomplete=True, incomplete_reason=reason)

    def finish_without_tools(self, completion: Completion) -> Result | None:
        text = completion.text
        if self.guards.observe_turn(text, 0):
            return self.result(no_output_stop_answer(self.turns), stop_reason="no_output")
        if not text.strip():
            self.append(Message(role="user", content=EMPTY_TURN_NUDGE))
            return None
        if ends_with_promise(text) and not self.promise_nudged:
            self.promise_nudged = True
            self.append(Message(role="user", content=promise_nudge()))
            return None
        if self.o.hooks:
            stop = self.o.hooks.dispatch("stop", {"text": text, "turns": self.turns})
            if stop.blocked and self.nudges < LIMITS.max_continue_nudges:
                self.nudges += 1
                self.append(Message(role="user", content=f"A stop hook ({stop.blocked_by}) asked you to continue: {' '.join(stop.context) or 'work remains'}"))
                return None
        if not self.o.require_completion_signal:
            return self.result(text)
        reason = self.incomplete_reason(text)
        if reason:
            return self.nudge(text, reason, continue_nudge(reason))
        if self.o.verify:
            verdict = verify(self.provider, self.objective, self.messages, self.ctx.state.get("plan", []))
            self.emit({"type": "verdict", "passed": verdict.passed, "reason": verdict.reason, "next_action": verdict.next_action})
            if not verdict.passed:
                return self.nudge(text, verdict.reason, continue_nudge(f"verifier: {verdict.reason}. Next: {verdict.next_action}"))
        return self.result(text)

    def child_policy(self, agent: Agent) -> Policy:
        parent = self.o.policy
        policy = Policy(parent.workspace, parent.mode, sandboxed=parent.sandboxed, allow_tools=parent.allow_tools,
                        deny_tools=parent.deny_tools, extra_dirs=parent.extra_dirs)
        policy.scope_to(agent.tools)
        policy.request_kind = parent.request_kind
        for name in parent.session_grants:
            policy.grant_session(name)
        for prefix in parent.prefix_grants:
            policy.grant_prefix(prefix)
        return policy

    def spawn(self, args: dict[str, Any]) -> ToolResult:
        o = self.o
        if o.depth >= LIMITS.delegate_depth:
            return ToolResult.error(f"Error: delegation depth {LIMITS.delegate_depth} reached; do this part yourself")
        task = str(args.get("task") or "").strip()
        if not task:
            return ToolResult.error("Error: task must not be empty")
        wanted = str(args.get("agent") or "").strip()
        agent = o.agents.get(wanted) if wanted else None
        if wanted and agent is None:
            return ToolResult.error(f"Error: unknown agent {wanted!r}; available: {', '.join(sorted(o.agents)) or 'none'}")
        profile: dict[str, Any] = {} if agent is None else {
            "system_prompt": f"{o.system_prompt}\n\n{agent_block(agent.prompt)}",
            "policy": self.child_policy(agent),
        }
        sid = o.session.create(cwd=str(o.workspace), model="", title=task, parent=o.session_id) if o.session else ""
        child_options = replace(
            o, **profile, history=[], session_id=sid, depth=o.depth + 1, on_ask_user=None, verify=False, require_completion_signal=True,
            max_turns=min(int(args.get("max_turns") or LIMITS.delegate_max_turns), LIMITS.delegate_max_turns),
            token_budget=max(int(args.get("budget_tokens") or LIMITS.delegate_budget_tokens), LIMITS.delegate_min_budget_tokens),
            on_event=(lambda event: o.on_event({**event, "child": sid})) if o.on_event else None,
        )
        self.emit({"type": "delegate", "child": sid, "task": task, "depth": o.depth + 1, "agent": wanted})
        child = _Run(self.provider, child_options)
        res = child.run(self.handoff(task, args))
        self.tokens_used += child.tokens_used
        self.cost_usd += child.cost_usd
        self.kept_out_tokens += child.tokens_used + child.kept_out_tokens
        self.saved_tokens += child.saved_tokens
        self.changed.update(child.changed)
        status = f"incomplete ({res.incomplete_reason})" if res.incomplete else "done"
        answer = clip(res.final_answer, LIMITS.delegate_answer_tokens * LIMITS.chars_per_token)
        label = f"{sid or 'child'} as {wanted}" if wanted else (sid or "child")
        head = f"[delegate {label}] {status}, {res.turns} turns, {child.tokens_used:,} tokens, ${child.cost_usd:.4f}"
        if child.changed:
            head += "\nchanged: " + ", ".join(sorted(child.changed))
        if child.refs:
            head += "\nresults it stored: " + ", ".join(f"§{ref}" for ref in child.refs[-LIMITS.delegate_refs_returned:])
        return ToolResult.success(f"{head}\n\n{answer}", changed_files=sorted(child.changed), meta={"full": res.final_answer})

    def handoff(self, task: str, args: dict[str, Any]) -> str:
        parts = [task]
        store = self.o.registry.observations
        for raw in args.get("refs") or []:
            found = store.load(str(raw).lstrip("§")) if store else None
            if found:
                parts.append(f"<result ref=\"§{found.ref}\" tool=\"{found.tool}\">\n{clip(found.body, LIMITS.delegate_handoff_tokens * LIMITS.chars_per_token)}\n</result>")
        if files := args.get("files"):
            parts.append("Start by reading: " + ", ".join(str(f) for f in files))
        return "\n\n".join(parts)

    def run_call(self, call: ToolCall) -> tuple[FailureOutcome, str]:
        res, denied = self.execute(call)
        self.changed.update(res.changed_files)
        self.loaded.update(res.meta.get("load_tools", []))
        if res.ok and res.meta.get("control"):
            self.control = str(res.meta["control"])
        if res.artifact:
            self.refs.append(res.artifact.ref)
        if res.diagnostics:
            self.saved_tokens += max(0, res.diagnostics.original_tokens - res.diagnostics.model_tokens)
        self.append(Message(role="tool", content=label_untrusted(call.name, res.output), tool_call_id=call.id, is_error=not res.ok))
        self.emit({"type": "tool_result", "id": call.id, "name": call.name, "ok": res.ok, "output": res.output, "changed_files": res.changed_files,
                   "display": asdict(res.display), "ref": res.artifact.ref if res.artifact else "",
                   "diagnostics": asdict(res.diagnostics) if res.diagnostics else {}})
        outcome = self.guards.observe_tool_result(call.name, not res.ok and not denied, res.output)
        if not outcome.hint:
            return outcome, ""
        tool = self.o.registry.get(call.name)
        return outcome, tool_failure_hint(call.name, json.dumps(tool.parameters if tool else {}), res.output)

    def abort_rest(self, rest: list[ToolCall]) -> None:
        for call in rest:
            self.append(Message(role="tool", content=ABORTED_TOOL_RESULT, tool_call_id=call.id, is_error=True))

    def turn_reminders(self, completion: Completion, call_count: int) -> list[str]:
        candidates = (calls_per_turn_reminder(call_count) if call_count > LIMITS.max_calls_per_turn else "",
                      self.guards.tool_only_reminder(completion.text, call_count),
                      self.guards.stale_plan_reminder(bool(pending_items(self.ctx.state))))
        return [reminder for reminder in candidates if reminder]

    def run_tools(self, completion: Completion) -> Result | None:
        calls = completion.tool_calls
        self.guards.observe_turn(completion.text, len(calls))
        followups: list[str] = []
        for index, call in enumerate(calls):
            if not call.name:
                self.append(Message(role="tool", content=DROPPED_TOOL_CALL_NOTICE, tool_call_id=call.id, is_error=True))
                continue
            if self.o.cancelled and self.o.cancelled():
                self.abort_rest(calls[index:])
                return self.stopped()
            if repeated := self.guards.observe_identical(call.name, call.arguments):
                followups.append(repeated)
            outcome, hint = self.run_call(call)
            if hint:
                followups.append(hint)
            if outcome.stop:
                self.abort_rest(calls[index + 1:])
                return self.result(tool_failure_stop_answer(call.name, outcome.count), stop_reason="tool_failure_loop")
            if self.control:
                self.abort_rest(calls[index + 1:])
                return self.result(self.messages[-1].content, stop_reason=self.control)
        for text in followups + self.turn_reminders(completion, len(calls)):
            self.append(Message(role="user", content=text))
        return None

    def account(self, usage: Usage) -> None:
        self.meter.observe(usage)
        cost = self.o.model_info.cost(usage) if self.o.model_info else 0.0
        self.tokens_used += usage.total
        self.cost_usd += cost
        self.emit({"type": "usage", "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                   "cache_read_tokens": usage.cache_read_tokens, "run_total": self.tokens_used,
                   "cost_usd": round(cost, LIMITS.usd_decimals), "run_cost_usd": round(self.cost_usd, LIMITS.usd_decimals),
                   "context_used": usage.input_tokens, "context_window": self.o.context_window,
                   "saved_tokens": self.saved_tokens, "kept_out_tokens": self.kept_out_tokens})

    def stopped(self) -> Result | None:
        if self.o.cancelled and self.o.cancelled():
            self.emit({"type": "cancelled", "turns": self.turns})
            return self.result("Stopped by the user.", incomplete=True, incomplete_reason="cancelled", stop_reason="cancelled")
        return None

    def budget_spent(self) -> Result | None:
        o = self.o
        if o.token_budget and self.tokens_used >= o.token_budget:
            reason = f"token budget of {o.token_budget} was reached after {self.tokens_used} tokens"
        elif o.budget_usd and self.cost_usd >= o.budget_usd:
            reason = f"spend budget of ${o.budget_usd:.2f} was reached after ${self.cost_usd:.4f}"
        else:
            return None
        self.emit({"type": "budget", "tokens": self.tokens_used, "cost_usd": round(self.cost_usd, LIMITS.usd_decimals), "reason": reason})
        return self.result(f"Stopped: the run's {reason}.", incomplete=True, incomplete_reason="budget reached", stop_reason="budget")

    def final_answer_after_max_turns(self) -> Result:
        self.append(Message(role="user", content=MAX_TURNS_FINAL_ANSWER_PROMPT))
        final = self.provider.complete(self.messages, [])
        self.append(Message(role="assistant", content=final.text))
        if final.text:
            self.emit({"type": "text", "text": final.text})
        headless = self.o.require_completion_signal
        return self.result(final.text or "Reached the turn limit without a final answer.",
                           incomplete=headless, incomplete_reason="turn limit reached" if headless else "", stop_reason="max_turns")

    def run(self, prompt: str) -> Result:
        o = self.o
        self.objective = prompt
        self.messages = [Message(role="system", content=o.system_prompt), *o.history]
        self.seqs = [0] * len(self.messages)
        self.persist("prompt", {"hash": prompt_hash(o.system_prompt), "tokens": approx_tokens(o.system_prompt), "text": o.system_prompt})
        self.append(Message(role="user", content=prompt, images=list(o.images)))
        if o.hooks:
            if o.session_start:
                for line in o.hooks.dispatch("sessionStart", {"session": o.session_id, "prompt": prompt}, "resume" if o.history else "startup").context:
                    self.append(Message(role="user", content=f"[hook] {line}"))
            for line in o.hooks.dispatch("userPrompt", {"session": o.session_id, "prompt": prompt}).context:
                self.append(Message(role="user", content=f"[hook] {line}"))
        for turn in range(max(1, o.max_turns)):
            self.turns = turn + 1
            if spent := self.budget_spent() or self.stopped():
                return spent
            exposed = o.registry.definitions(o.policy.visible, self.loaded)
            self.maybe_compact(exposed)
            completion = self.complete(exposed)
            self.account(completion.usage)
            self.append(Message(role="assistant", content=completion.text, tool_calls=list(completion.tool_calls)))
            if completion.text:
                self.emit({"type": "text", "text": completion.text})
            outcome = self.run_tools(completion) if completion.tool_calls else self.finish_without_tools(completion)
            if outcome is not None:
                return outcome
        return self.final_answer_after_max_turns()


def run(prompt: str, provider: Provider, options: Options) -> Result:
    return _Run(provider, options).run(prompt)
