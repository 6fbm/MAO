"""Text-based tool calling for models without a native tool API.

The request is rewritten so tools are described in the system prompt and
tool results arrive as user messages; the response is scanned for
```` ```tool ```` blocks that are turned back into neutral ``ToolCall`` objects.
"""

from __future__ import annotations

import json
import re

from mao.core.jsonutil import extract_json_object
from mao.core.types import ChatMessage, CompletionRequest, CompletionResponse, FinishReason, MessageRole, ToolCall, new_id

_TOOL_BLOCK_RE = re.compile(r"```[ \t]*tool[ \t]*\r?\n(.*?)```", re.S)
_JSON_BLOCK_RE = re.compile(r"```[ \t]*json[ \t]*\r?\n(.*?)```", re.S)


def tools_instructions(request: CompletionRequest) -> str:
    lines = [
        "## Tools",
        "You can call tools. To call a tool, reply with one or more blocks in exactly this format and nothing else:",
        "```tool",
        '{"name": "<tool name>", "arguments": {"<param>": "<value>"}}',
        "```",
        "You will receive the results in the next message. When you are finished, reply normally without any tool block.",
        "",
        "Available tools:",
    ]
    for tool in request.tools:
        lines.append(f"- {tool.name}: {tool.description}")
        lines.append(f"  parameters (JSON schema): {json.dumps(tool.parameters, ensure_ascii=False)}")
    return "\n".join(lines)


def to_prompt_request(request: CompletionRequest) -> CompletionRequest:
    if not request.tools:
        return request
    system = ((request.system or "") + "\n\n" + tools_instructions(request)).strip()
    messages: list[ChatMessage] = []
    for message in request.messages:
        if message.role is MessageRole.ASSISTANT:
            blocks = [
                "```tool\n" + json.dumps({"name": c.name, "arguments": c.arguments}, ensure_ascii=False) + "\n```"
                for c in message.tool_calls
            ]
            content = "\n".join(part for part in [message.content, *blocks] if part)
            messages.append(ChatMessage.assistant(content or "(no output)"))
        elif message.role is MessageRole.TOOL:
            status = "ERROR" if message.is_error else "OK"
            messages.append(ChatMessage.user(f"[Tool result {status}] {message.name} ({message.tool_call_id}):\n{message.content}"))
        else:
            messages.append(ChatMessage.user(message.content))
    merged: list[ChatMessage] = []
    for message in messages:
        if merged and merged[-1].role is message.role is MessageRole.USER:
            merged[-1] = ChatMessage.user(merged[-1].content + "\n\n" + message.content)
        else:
            merged.append(message)
    return request.model_copy(update={"system": system, "messages": merged, "tools": []})


def parse_prompt_response(response: CompletionResponse, tool_names: set[str]) -> CompletionResponse:
    text = response.message.content or ""
    calls: list[ToolCall] = []
    matched_spans: list[tuple[int, int]] = []
    for pattern, strict in ((_TOOL_BLOCK_RE, False), (_JSON_BLOCK_RE, True)):
        for match in pattern.finditer(text):
            obj = extract_json_object(match.group(1))
            if not obj or not isinstance(obj.get("name"), str):
                continue
            if strict and obj["name"] not in tool_names:
                continue
            arguments = obj.get("arguments", obj.get("args", {}))
            calls.append(
                ToolCall(
                    id=new_id("ptool_"),
                    name=obj["name"],
                    arguments=arguments if isinstance(arguments, dict) else {},
                    parse_error=None if isinstance(arguments, dict) else "Arguments must be an object",
                )
            )
            matched_spans.append(match.span())
        if calls:
            break
    if not calls:
        return response
    remaining = text
    for start, end in sorted(matched_spans, reverse=True):
        remaining = remaining[:start] + remaining[end:]
    response.message = ChatMessage.assistant(remaining.strip(), calls)
    response.finish_reason = FinishReason.TOOL_CALLS
    return response
