from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from superclaw.settings import LIMITS

_PROMPTS = Path(__file__).parent / "prompts"
SCRIPTS = (("typecheck", "typecheck"), ("test", "tests"), ("build", "build"), ("lint", "lint"))
LOCKFILES = (("bun", ("bun.lock", "bun.lockb")), ("pnpm", ("pnpm-lock.yaml",)), ("yarn", ("yarn.lock",)), ("npm", ("package-lock.json",)))


@dataclass(frozen=True)
class Check:
    id: str
    name: str
    command: list[str]
    kind: str


@dataclass
class Result:
    check: Check
    status: str
    exit_code: int
    tail: list[str]
    duration_ms: int

    @property
    def ok(self) -> bool:
        return self.status == "passed"


@dataclass
class Report:
    root: Path
    results: list[Result] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    @property
    def failed(self) -> list[Result]:
        return [r for r in self.results if not r.ok]


def _package_manager(root: Path, declared: str) -> str:
    name = declared.split("@", maxsplit=1)[0].strip().lower()
    for manager, locks in LOCKFILES:
        if name == manager or any((root / lock).exists() for lock in locks):
            return manager
    return "npm"


def _package_checks(root: Path) -> list[Check]:
    try:
        package = json.loads((root / "package.json").read_text())
    except (OSError, ValueError):
        return []
    scripts = package.get("scripts") or {} if isinstance(package, dict) else {}
    manager = _package_manager(root, str(package.get("packageManager") or ""))
    return [Check(f"{manager}.{script}", f"{manager.capitalize()} {label}", [manager, "run", script], "test" if script == "test" else script)
            for script, label in SCRIPTS if str(scripts.get(script) or "").strip()]


def _has_pytest(root: Path) -> bool:
    if (root / "pytest.ini").is_file():
        return True
    pyproject = root / "pyproject.toml"
    if pyproject.is_file() and "[tool.pytest" in pyproject.read_text(errors="replace"):
        return True
    return (root / "tests").is_dir() and (pyproject.is_file() or (root / "setup.py").is_file())


def detect(root: Path) -> list[Check]:
    root = Path(root).resolve()
    checks: list[Check] = []
    if (root / "go.mod").is_file():
        checks.append(Check("go.test", "Go tests", ["go", "test", "./..."], "test"))
    checks += _package_checks(root)
    if _has_pytest(root):
        checks.append(Check("python.pytest", "Python pytest", ["python", "-m", "pytest"], "test"))
    if (root / "Cargo.toml").is_file():
        checks.append(Check("cargo.test", "Cargo tests", ["cargo", "test"], "test"))
    return checks


def run(root: Path, checks: list[Check], only: tuple[str, ...] = (), timeout_s: int = LIMITS.verify_timeout_s) -> Report:
    report = Report(Path(root).resolve())
    for check in checks:
        if only and check.id not in only:
            continue
        started = time.monotonic()
        try:
            done = subprocess.run(check.command, cwd=report.root, capture_output=True, text=True, timeout=timeout_s, check=False)
            output, code, status = done.stdout + done.stderr, done.returncode, "passed" if done.returncode == 0 else "failed"
        except subprocess.TimeoutExpired as e:
            output, code, status = (e.stdout or b"").decode("utf-8", errors="replace") + (e.stderr or b"").decode("utf-8", errors="replace"), -1, "timed_out"
        except OSError as e:
            output, code, status = str(e), -1, "error"
        tail = [line for line in output.strip().splitlines() if line.strip()][-LIMITS.verify_output_lines:]
        report.results.append(Result(check, status, code, tail, int((time.monotonic() - started) * 1000)))
    return report


def remediation_prompt(report: Report) -> str:
    blocks = [f"<check id=\"{r.check.id}\" command=\"{' '.join(r.check.command)}\" status=\"{r.status}\" exit=\"{r.exit_code}\">\n"
              + "\n".join(r.tail) + "\n</check>" for r in report.failed]
    return (_PROMPTS / "verify.md").read_text().strip() + "\n\n" + "\n\n".join(blocks)


def lines(report: Report) -> list[str]:
    out = []
    for r in report.results:
        mark = "pass" if r.ok else r.status
        out.append(f"[{mark}] {r.check.name}: {' '.join(r.check.command)} ({r.duration_ms} ms)")
        if not r.ok:
            out += [f"    {line}" for line in r.tail]
    return out


def as_json(report: Report) -> dict:
    return {"root": str(report.root), "ok": report.ok,
            "results": [{"id": r.check.id, "name": r.check.name, "command": r.check.command, "kind": r.check.kind, "status": r.status,
                         "exitCode": r.exit_code, "durationMs": r.duration_ms, "tail": r.tail} for r in report.results]}
