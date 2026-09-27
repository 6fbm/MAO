"""Executes tool calls: validation, permissions, events, error isolation, truncation."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from mao.core.errors import OperationCancelled, PermissionDeniedError, ToolError
from mao.core.events import ToolEvent
from mao.core.text import one_line, truncate_middle
from mao.core.types import RunMode, ToolCall, ToolSpec
from mao.security.permissions import CAPABILITY_LABELS, missing_capabilities
from mao.security.redaction import Redactor
from mao.tools.base import Tool, ToolContext, ToolResult
from mao.tools.registry import ToolRegistry

log = logging.getLogger("mao.tools")

_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}
_LARGE_ARGS = {"content", "new_text", "old_text", "question"}


def validate_arguments(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    properties = schema.get("properties") or {}
    for name in schema.get("required") or []:
        if name not in args or args[name] is None:
            return f"Required parameter missing: '{name}'"
    for name, value in args.items():
        spec = properties.get(name)
        if spec is None or value is None:
            continue
        expected = spec.get("type")
        types = expected if isinstance(expected, list) else [expected]
        allowed: tuple[type, ...] = ()
        for type_name in types:
            allowed += _TYPE_CHECKS.get(type_name, ())
        if allowed:
            if isinstance(value, bool) and bool not in allowed:
                return f"Parameter '{name}' has the wrong type (expected {expected})"
            if not isinstance(value, allowed):
                return f"Parameter '{name}' has the wrong type (expected {expected}, got {type(value).__name__})"
        if "enum" in spec and value not in spec["enum"]:
            return f"Parameter '{name}' must be one of {spec['enum']}"
    return None


def summarize_arguments(args: dict[str, Any]) -> str:
    parts = []
    for key, value in args.items():
        if key in _LARGE_ARGS and isinstance(value, str) and len(value) > 60:
            parts.append(f"{key}=<{len(value)} characters>")
        else:
            parts.append(f"{key}={value!r}")
    return one_line(", ".join(parts), 160)


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, *, redactor: Redactor, max_result_chars: int = 12_000) -> None:
        self.registry = registry
        self.redactor = redactor
        self.max_result_chars = max_result_chars

    def tools_for(self, ctx: ToolContext, groups: list[str]) -> list[Tool]:
        return self.registry.available_for(
            groups=groups,
            permissions=ctx.permissions,
            mode=ctx.mode,
            enabled_groups=ctx.tools_config.enabled_groups,
        )

    def specs_for(self, ctx: ToolContext, groups: list[str]) -> list[ToolSpec]:
        return [tool.spec() for tool in self.tools_for(ctx, groups)]

    async def execute(self, call: ToolCall, ctx: ToolContext, allowed: set[str]) -> ToolResult:
        tool = self.registry.get(call.name)
        if tool is None or call.name not in allowed:
            return ToolResult.failure(
                f"Unknown or forbidden tool: '{call.name}'. Available: {', '.join(sorted(allowed)) or 'none'}"
            )
        if call.parse_error:
            return ToolResult.failure(f"Invalid arguments for {call.name}: {call.parse_error}")
        problem = validate_arguments(tool.parameters, call.arguments)
        if problem:
            return ToolResult.failure(f"{call.name}: {problem}")
        missing = missing_capabilities(ctx.permissions, tool.required)
        if missing:
            labels = ", ".join(CAPABILITY_LABELS[c] for c in sorted(missing, key=lambda c: c.value))
            return ToolResult.failure(f"No permission for {call.name}: needs {labels}")
        if ctx.mode is RunMode.PLAN and tool.mutating and not tool.plan_allowed:
            return ToolResult.failure(f"{call.name} is not allowed in PLAN mode (no lasting changes).")

        summary = summarize_arguments(call.arguments)
        ctx.bus.publish(ToolEvent(stage="start", agent=ctx.agent_name, tool=call.name, summary=summary))
        started = time.monotonic()
        try:
            result = await tool.run(call.arguments, ctx)
        except (OperationCancelled, asyncio.CancelledError):
            raise
        except PermissionDeniedError as exc:
            result = ToolResult.failure(f"Rejected: {exc}")
        except ToolError as exc:
            result = ToolResult.failure(str(exc))
        except Exception as exc:  # noqa: BLE001 - a tool bug must not kill the agent
            log.exception("tool %s failed", call.name)
            result = ToolResult.failure(f"Internal error in {call.name} ({type(exc).__name__}): {exc}")

        content = self.redactor.redact(result.content)
        content, truncated = truncate_middle(content, self.max_result_chars)
        result.content = content
        if truncated:
            result.data["truncated"] = True
        ctx.bus.publish(
            ToolEvent(
                stage="end",
                agent=ctx.agent_name,
                tool=call.name,
                summary=one_line(content.splitlines()[0] if content else "", 140),
                ok=result.ok,
                duration_s=round(time.monotonic() - started, 3),
                detail=content[:2_000],
            )
        )
        return result
