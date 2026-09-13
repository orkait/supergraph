from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from superclaw.intent import Kind
from superclaw.tools import PathEscapes, Permission, SideEffect, Tool, jail


class Mode(str, Enum):
    ASK = "ask"
    AUTO = "auto"
    PLAN = "plan"
    UNSAFE = "unsafe"


CYCLE_MODES = (Mode.ASK, Mode.AUTO, Mode.PLAN)


def next_mode(mode: Mode) -> Mode:
    if mode not in CYCLE_MODES:
        return CYCLE_MODES[0]
    return CYCLE_MODES[(CYCLE_MODES.index(mode) + 1) % len(CYCLE_MODES)]


class Action(str, Enum):
    ALLOW = "allow"
    PROMPT = "prompt"
    DENY = "deny"


@dataclass
class Risk:
    level: str
    categories: list[str] = field(default_factory=list)


@dataclass
class Decision:
    action: Action
    reason: str
    risk: Risk
    escalated: bool = False
    network: bool = False


@dataclass
class Classification:
    categories: list[str]
    segments: list[list[str]]


@dataclass
class ShellRequest:
    cls: Classification
    wants_host: bool
    justification: str
    outside: list[str]
    risk: Risk
    escalated: bool
    network: bool

    def decide(self, action: Action, reason: str) -> Decision:
        return Decision(action, reason, self.risk, self.escalated, self.network)

    def prompt_reason(self) -> str:
        if self.wants_host:
            return f"runs outside the sandbox: {self.justification}"
        if "destructive" in self.cls.categories:
            return "destructive shell command requires approval"
        if self.outside:
            return f"write access outside the workspace: {', '.join(self.outside)}"
        if self.network:
            return "network access requires approval"
        return ""


_SEPARATORS = {";", "|", "||", "&&", "&", "|&"}
_WRAPPERS = {"env", "nohup", "time", "command", "exec", "nice", "stdbuf"}
_PRIVILEGED = {"sudo", "doas", "su"}
_DESTRUCTIVE = {"mkfs", "fdisk", "shred", "dd", "parted", "wipefs"}
_NETWORK = {"curl", "wget", "ssh", "scp", "sftp", "rsync", "nc", "ncat", "netcat", "telnet", "ftp", "gh", "xh", "http", "https", "httpie", "pipx"}
_INTERACTIVE = {"vim", "vi", "nvim", "nano", "emacs", "less", "more", "top", "htop", "btop", "man"}
_REPLS = {"python", "python3", "node", "irb", "ipython", "bpython", "psql", "mysql", "sqlite3", "bash", "sh", "zsh", "fish"}
_RECURSIVE_FLAGS = {"-r", "-R", "--recursive"}
_GIT_NETWORK = {"push", "pull", "fetch", "clone", "ls-remote"}
_DOCKER_DESTRUCTIVE = {"system prune", "volume rm", "container prune", "image prune", "volume prune"}
_INSTALL_SUBCOMMANDS: dict[str, set[str]] = {
    "pip": {"install", "download"},
    "pip3": {"install", "download"},
    "npm": {"install", "i", "ci", "add", "update"},
    "pnpm": {"install", "i", "add", "update"},
    "yarn": {"install", "add", "upgrade"},
    "bun": {"install", "i", "add", "update"},
    "cargo": {"install", "add", "fetch", "update"},
    "go": {"get", "install"},
    "apt": {"install", "update", "upgrade"},
    "apt-get": {"install", "update", "upgrade"},
    "dnf": {"install", "update", "upgrade"},
    "yum": {"install", "update", "upgrade"},
    "brew": {"install", "upgrade", "update"},
    "pacman": {"-S", "-Syu"},
    "gem": {"install"},
}
_LEVEL = {SideEffect.NONE: "low", SideEffect.READ: "low", SideEffect.WRITE: "medium", SideEffect.SHELL: "medium", SideEffect.NETWORK: "high"}


def _tokens(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError:
        return command.split()


def _segments(tokens: list[str]) -> list[list[str]]:
    out: list[list[str]] = [[]]
    for tok in tokens:
        if tok in _SEPARATORS:
            out.append([])
        elif tok not in {"(", ")"}:
            out[-1].append(tok)
    return [s for s in out if s]


def _strip_prefix(seg: list[str]) -> tuple[list[str], bool]:
    privileged = False
    i = 0
    while i < len(seg):
        tok = seg[i]
        if tok in _PRIVILEGED:
            privileged = True
            i += 1
            while i < len(seg) and seg[i].startswith("-"):
                i += 1
        elif tok in _WRAPPERS or ("=" in tok and not tok.startswith("-")):
            i += 1
        else:
            break
    return seg[i:], privileged


def _program(seg: list[str]) -> tuple[str, list[str]]:
    prog = Path(seg[0]).name
    args = seg[1:]
    if prog.startswith("python") and args[:2] in (["-m", "pip"], ["-m", "pip3"]):
        return "pip", args[2:]
    return prog, args


def _subcommand(args: list[str]) -> str:
    return next((a for a in args if not a.startswith("-")), "")


def _has_recursive_flag(flags: set[str]) -> bool:
    return any(f in _RECURSIVE_FLAGS or (f.startswith("-") and not f.startswith("--") and ("r" in f or "R" in f)) for f in flags)


def _install_categories(prog: str, args: list[str]) -> set[str]:
    if prog == "uv":
        return {"network"} if args[:2] == ["pip", "install"] or args[:1] in (["add"], ["sync"]) else set()
    if prog not in _INSTALL_SUBCOMMANDS:
        return set()
    sub = _subcommand(args)
    if (prog == "yarn" and not sub) or sub in _INSTALL_SUBCOMMANDS[prog]:
        return {"network"}
    return set()


def _git_categories(args: list[str], flags: set[str]) -> set[str]:
    sub = _subcommand(args)
    cats: set[str] = set()
    if sub in _GIT_NETWORK:
        cats.add("network")
    if (sub == "push" and flags & {"-f", "--force", "--force-with-lease"}) or (sub == "reset" and "--hard" in flags) \
            or sub == "clean" or (sub == "branch" and "-D" in flags):
        cats.add("destructive")
    if sub == "rebase" and flags & {"-i", "--interactive"}:
        cats.add("interactive")
    return cats


def _docker_categories(args: list[str]) -> set[str]:
    words = [a for a in args[:2] if not a.startswith("-")]
    if words[:1] in (["rm"], ["rmi"]) or " ".join(words) in _DOCKER_DESTRUCTIVE:
        return {"destructive"}
    return set()


def _classify_segment(seg: list[str]) -> set[str]:
    seg, privileged = _strip_prefix(seg)
    cats: set[str] = {"destructive"} if privileged else set()
    if not seg:
        return cats
    prog, args = _program(seg)
    flags = {a for a in args if a.startswith("-")}
    if prog in _INTERACTIVE or (prog in _REPLS and not args):
        cats.add("interactive")
    if prog in _NETWORK:
        cats.add("network")
    if prog.startswith("mkfs") or prog in _DESTRUCTIVE:
        cats.add("destructive")
    if (prog == "rm" and _has_recursive_flag(flags)) or (prog in {"chmod", "chown"} and flags & {"-R", "--recursive"}):
        cats.add("destructive")
    cats |= _install_categories(prog, args)
    if prog == "git":
        cats |= _git_categories(args, flags)
    if prog == "docker":
        cats |= _docker_categories(args)
    return cats


def classify_command(command: str) -> Classification:
    segments = _segments(_tokens(command))
    cats: set[str] = set()
    for seg in segments:
        cats |= _classify_segment(seg)
    return Classification([c for c in ("interactive", "destructive", "network") if c in cats], segments)


_UNGRANTABLE = {"rm", "sudo", "doas", "su", "bash", "sh", "zsh", "fish", "node", "perl", "ruby", "eval", "exec", "xargs", "env"}


def validate_prefix(prefix: list[str], command: str) -> str:
    if not prefix or any(not isinstance(p, str) or not p for p in prefix):
        return "prefix_rule must be a non-empty list of tokens"
    head = Path(prefix[0]).name
    if head in _UNGRANTABLE or head.startswith("python"):
        return f"a prefix starting with {head!r} is too broad to remember"
    if len(prefix) == 1:
        return "a single-token prefix is too broad to remember"
    if "<<" in command:
        return "commands with a heredoc cannot be remembered by prefix"
    segments = _segments(_tokens(command))
    if not segments or _strip_prefix(segments[0])[0][:len(prefix)] != prefix:
        return "prefix_rule must match the start of the command"
    return ""


class Policy:
    def __init__(self, workspace: Path, mode: Mode = Mode.ASK, sandboxed: bool = False) -> None:
        self.workspace = Path(workspace)
        self.mode = mode
        self.sandboxed = sandboxed
        self.request_kind = Kind.CHANGE
        self._session_grants: set[str] = set()
        self._prefix_grants: list[list[str]] = []

    @property
    def session_grants(self) -> list[str]:
        return sorted(self._session_grants)

    @property
    def prefix_grants(self) -> list[list[str]]:
        return [list(p) for p in self._prefix_grants]

    def grant_session(self, tool_name: str) -> None:
        self._session_grants.add(tool_name)

    def grant_prefix(self, prefix: list[str]) -> None:
        self._prefix_grants.append(list(prefix))

    def _kind_blocks(self, se: SideEffect) -> str:
        if self.request_kind == Kind.ANSWER and se in (SideEffect.WRITE, SideEffect.SHELL, SideEffect.NETWORK):
            return "the request was classified as answer, which does not authorize writes, shell or network; ask the user for a change"
        if self.request_kind == Kind.DIAGNOSE and se == SideEffect.WRITE:
            return "the request was classified as diagnose; report the cause, do not implement the fix unless the user asks"
        return ""

    def visible(self, tool: Tool) -> bool:
        if tool.safety.permission == Permission.DENY:
            return False
        if self.mode == Mode.PLAN:
            return tool.safety.side_effect in (SideEffect.NONE, SideEffect.READ)
        return not self._kind_blocks(tool.safety.side_effect)

    def _prefix_covers(self, segments: list[list[str]]) -> bool:
        if not segments or not self._prefix_grants:
            return False
        return all(any(_strip_prefix(seg)[0][:len(p)] == p for p in self._prefix_grants) for seg in segments)

    def _path_block(self, args: dict[str, Any]) -> str:
        for key in ("path", "cwd"):
            value = args.get(key)
            if value:
                try:
                    jail(self.workspace, str(value))
                except PathEscapes as e:
                    return str(e)
        return ""

    def _shell_request(self, args: dict[str, Any]) -> ShellRequest:
        cls = classify_command(str(args.get("command") or ""))
        extra = args.get("additional_permissions") or {}
        wants_host = args.get("sandbox_permissions") == "require_escalated"
        network = "network" in cls.categories or bool(extra.get("network"))
        outside = [p for p in extra.get("paths") or [] if not self._inside(p)]
        categories = ["shell", *cls.categories, *(["out_of_workspace"] if outside else [])]
        escalated = wants_host or not self.sandboxed
        level = "critical" if "destructive" in cls.categories else "high" if network or escalated or outside else "medium"
        return ShellRequest(cls, wants_host, str(args.get("justification") or "").strip(), outside, Risk(level, categories), escalated, network)

    def _evaluate_shell(self, tool: Tool, args: dict[str, Any]) -> Decision:
        req = self._shell_request(args)
        if "interactive" in req.cls.categories:
            return Decision(Action.DENY, "interactive programs hang the agent; use a non-interactive form", Risk("high", req.cls.categories))
        if self._prefix_covers(req.cls.segments):
            return req.decide(Action.ALLOW, "approved command prefix")
        if self.mode == Mode.UNSAFE:
            return req.decide(Action.ALLOW, "unsafe mode")
        if req.wants_host and not req.justification:
            return Decision(Action.DENY, "require_escalated needs a justification", req.risk)
        prompt_reason = req.prompt_reason()
        if prompt_reason:
            return req.decide(Action.PROMPT, prompt_reason)
        if tool.name in self._session_grants:
            return req.decide(Action.ALLOW, "session grant")
        if self.mode == Mode.AUTO:
            return req.decide(Action.ALLOW, "sandboxed workspace shell auto-allowed") if self.sandboxed \
                else req.decide(Action.PROMPT, "no sandbox backend; approve to run unsandboxed")
        return req.decide(Action.PROMPT, tool.safety.reason)

    def _inside(self, path: str) -> bool:
        try:
            jail(self.workspace, path)
        except PathEscapes:
            return False
        return True

    def evaluate(self, tool: Tool, args: dict[str, Any]) -> Decision:
        safety = tool.safety
        se = safety.side_effect
        risk = Risk(_LEVEL[se], [se.value] if se != SideEffect.NONE else [])
        if safety.permission == Permission.DENY:
            return Decision(Action.DENY, safety.reason, risk)
        if block := self._path_block(args):
            return Decision(Action.DENY, block, risk)
        if se in (SideEffect.NONE, SideEffect.READ):
            return Decision(Action.ALLOW, "read-only", risk)
        if self.mode == Mode.PLAN:
            return Decision(Action.DENY, "plan mode is read-only", risk)
        if blocked := self._kind_blocks(se):
            return Decision(Action.DENY, blocked, risk)
        if se == SideEffect.SHELL:
            return self._evaluate_shell(tool, args)
        if self.mode == Mode.UNSAFE:
            return Decision(Action.ALLOW, "unsafe mode", risk)
        if tool.name in self._session_grants:
            return Decision(Action.ALLOW, "session grant", risk)
        if se == SideEffect.WRITE and self.mode == Mode.AUTO:
            return Decision(Action.ALLOW, "workspace write auto-allowed", risk)
        return Decision(Action.PROMPT, safety.reason, risk)
