"""Test runner tool: detects the test command, runs it and parses the summary."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from mao.core.errors import ToolError
from mao.core.text import truncate_middle
from mao.security.permissions import Capability
from mao.tools.base import Tool, ToolContext, ToolResult, object_schema
from mao.tools.terminal import authorize_command, resolve_shell, run_shell_command, scrubbed_environment
from mao.workspace.workspace import detect_test_command


class TestRunSummary(BaseModel):
    __test__ = False  # not a pytest test class

    command: str
    exit_code: int | None
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration_s: float = 0.0
    timed_out: bool = False
    framework: str | None = None
    output_tail: str = ""

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.failed == 0 and self.errors == 0

    def headline(self) -> str:
        state = "PASSED" if self.success else ("TIMEOUT" if self.timed_out else "FAILED")
        return (
            f"Tests {state}: {self.passed} passed, {self.failed} failed, "
            f"{self.errors} errors, {self.skipped} skipped (exit {self.exit_code}, {self.duration_s:.1f}s)"
        )


def _num(pattern: str, text: str) -> int:
    matches = re.findall(pattern, text)
    return int(matches[-1]) if matches else 0


_PYTEST_SUMMARY_RE = re.compile(
    r"^[=\s]*((?:\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|warnings?|deselected|rerun)(?:, )?)+) in [\d.]+s",
    re.M,
)


def parse_test_output(output: str) -> dict[str, Any]:
    text = output[-20_000:]
    pytest_summary = _PYTEST_SUMMARY_RE.findall(text)
    if pytest_summary or re.search(r"^[=\s]*no tests ran in [\d.]+s", text, re.M):
        line = pytest_summary[-1] if pytest_summary else ""
        return {
            "framework": "pytest",
            "passed": _num(r"(\d+) passed", line),
            "failed": _num(r"(\d+) failed", line),
            "errors": _num(r"(\d+) errors?", line),
            "skipped": _num(r"(\d+) skipped", line),
        }
    jest = re.search(r"Tests:\s+(.*?)(\d+) total", text)
    if jest:
        part = jest.group(1)
        return {"framework": "jest", "passed": _num(r"(\d+) passed", part), "failed": _num(r"(\d+) failed", part), "errors": 0, "skipped": _num(r"(\d+) skipped", part)}
    cargo = re.findall(r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored", text)
    if cargo:
        return {"framework": "cargo", "passed": sum(int(c[0]) for c in cargo), "failed": sum(int(c[1]) for c in cargo), "errors": 0, "skipped": sum(int(c[2]) for c in cargo)}
    dotnet = re.search(r"Failed:\s*(\d+),\s*Passed:\s*(\d+),\s*Skipped:\s*(\d+)", text)
    if dotnet:
        return {"framework": "dotnet", "passed": int(dotnet.group(2)), "failed": int(dotnet.group(1)), "errors": 0, "skipped": int(dotnet.group(3))}
    if re.search(r"^(ok|FAIL)\s+\S+", text, re.M):
        return {"framework": "go", "passed": len(re.findall(r"^--- PASS", text, re.M)), "failed": len(re.findall(r"^--- FAIL", text, re.M)), "errors": 0, "skipped": len(re.findall(r"^--- SKIP", text, re.M))}
    unittest = re.search(r"Ran (\d+) tests?", text)
    if unittest:
        failures = _num(r"failures=(\d+)", text)
        errors = _num(r"errors=(\d+)", text)
        return {"framework": "unittest", "passed": int(unittest.group(1)) - failures - errors, "failed": failures, "errors": errors, "skipped": _num(r"skipped=(\d+)", text)}
    return {"framework": None, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}


async def execute_tests(ctx: ToolContext, command: str | None = None, timeout_s: int | None = None) -> TestRunSummary:
    root = ctx.sandbox.root
    resolved = command or ctx.tools_config.tests.command or detect_test_command(root)
    if not resolved:
        raise ToolError("No test command detected. Please pass 'command' or configure tools.tests.command.")
    await authorize_command(ctx, resolved, ".")
    terminal = ctx.tools_config.terminal
    outcome = await run_shell_command(
        resolved,
        cwd=root,
        shell=resolve_shell(terminal.shell),
        timeout_s=timeout_s or ctx.tools_config.tests.timeout_s,
        env=scrubbed_environment(ctx.permissions_config.scrub_env_patterns, terminal.env_passthrough),
    )
    combined = outcome.stdout + ("\n" + outcome.stderr if outcome.stderr else "")
    parsed = parse_test_output(combined)
    summary = TestRunSummary(
        command=resolved,
        exit_code=outcome.exit_code,
        duration_s=outcome.duration_s,
        timed_out=outcome.timed_out,
        output_tail=truncate_middle(combined.strip(), terminal.max_output_chars)[0],
        **parsed,
    )
    if summary.exit_code not in (0, None) and summary.failed == 0 and summary.errors == 0:
        summary.errors = 1  # non-zero exit without parsable failures (collection error, crash)
    if ctx.on_test_result is not None:
        ctx.on_test_result(summary)
    return summary


class RunTestsTool(Tool):
    name = "run_tests"
    group = "tests"
    description = "Run the project's test suite (auto-detected: pytest, npm test, cargo, go, dotnet) or a given test command."
    parameters = object_schema(
        {
            "command": {"type": "string", "description": "Optional explicit test command"},
            "timeout_s": {"type": "integer"},
        }
    )
    required = frozenset({Capability.EXECUTE})
    mutating = True
    plan_allowed = False
    parallel_safe = False

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        summary = await execute_tests(ctx, args.get("command"), args.get("timeout_s"))
        return ToolResult(
            ok=summary.success,
            content=f"{summary.headline()}\n$ {summary.command}\n\n{summary.output_tail}",
            data=summary.model_dump(),
        )


TOOLS = [RunTestsTool]
