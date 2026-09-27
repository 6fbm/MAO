"""Agent-to-agent collaboration tools: messages, shared findings, blackboard, consultations."""

from __future__ import annotations

from typing import Any

from mao.security.permissions import Capability
from mao.tools.base import Tool, ToolContext, ToolResult, object_schema

MESSAGE_KINDS = ["info", "question", "answer", "proposal", "critique", "warning"]
FINDING_KINDS = ["finding", "risk", "research", "recommendation"]


def _no_hub() -> ToolResult:
    return ToolResult.failure("Communication is not available in this context")


class SendMessageTool(Tool):
    name = "send_message"
    group = "collaboration"
    description = "Send a message to another agent (by name or role) or to 'all'. Use for handovers, warnings and proposals."
    parameters = object_schema(
        {
            "to": {"type": "string", "description": "Agent name, role name or 'all'"},
            "content": {"type": "string"},
            "kind": {"type": "string", "enum": MESSAGE_KINDS},
        },
        ["to", "content"],
    )
    required = frozenset()

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.collaboration is None:
            return _no_hub()
        message_id = ctx.collaboration.send_message(
            ctx.agent_name, [str(args["to"])], str(args.get("kind") or "info"), str(args["content"]), ctx.node_id
        )
        return ToolResult.success(f"Message {message_id} sent to {args['to']}.")


class PostFindingTool(Tool):
    name = "post_finding"
    group = "collaboration"
    description = (
        "Publish a finding, risk, research result or recommendation to the shared blackboard so other agents "
        "can use it. Include file paths and source URLs as evidence."
    )
    parameters = object_schema(
        {
            "title": {"type": "string"},
            "content": {"type": "string"},
            "kind": {"type": "string", "enum": FINDING_KINDS},
            "sources": {"type": "array", "items": {"type": "string"}, "description": "URLs"},
            "files": {"type": "array", "items": {"type": "string"}},
            "importance": {"type": "integer", "description": "1 (low) - 5 (critical)"},
        },
        ["title", "content"],
    )
    required = frozenset()

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.collaboration is None:
            return _no_hub()
        entry_id = ctx.collaboration.post_finding(
            ctx.agent_name,
            str(args["title"]),
            str(args["content"]),
            kind=str(args.get("kind") or "finding"),
            sources=[str(s) for s in args.get("sources") or []],
            files=[str(f) for f in args.get("files") or []],
            importance=max(1, min(int(args.get("importance") or 3), 5)),
            node_id=ctx.node_id,
        )
        return ToolResult.success(f"Entry {entry_id} published.")


class ReadBoardTool(Tool):
    name = "read_board"
    group = "collaboration"
    description = "Read shared blackboard entries (findings, research, decisions, results) in full, optionally filtered."
    parameters = object_schema(
        {
            "query": {"type": "string", "description": "Keywords or an entry id"},
            "kind": {"type": "string", "description": "finding, risk, research, recommendation, decision, result, workspace"},
            "limit": {"type": "integer"},
        }
    )
    required = frozenset()

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.collaboration is None:
            return _no_hub()
        text = ctx.collaboration.read_board(args.get("query"), args.get("kind"), max(1, min(int(args.get("limit") or 5), 20)))
        return ToolResult.success(text)


class ConsultAgentTool(Tool):
    name = "consult_agent"
    group = "collaboration"
    description = (
        "Ask another agent (by name or role) a focused question and wait for the answer. Costs tokens – use it "
        "only when you need expertise or verification you cannot get otherwise."
    )
    parameters = object_schema({"agent": {"type": "string"}, "question": {"type": "string"}}, ["agent", "question"])
    required = frozenset({Capability.CONSULT})
    parallel_safe = False

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.collaboration is None:
            return _no_hub()
        if ctx.consult_depth >= 1:
            return ToolResult.failure("A consulted agent must not consult further agents.")
        answer = await ctx.collaboration.consult(ctx.agent_name, str(args["agent"]), str(args["question"]), ctx.node_id, ctx.consult_depth)
        return ToolResult.success(answer)


TOOLS = [SendMessageTool, PostFindingTool, ReadBoardTool, ConsultAgentTool]
