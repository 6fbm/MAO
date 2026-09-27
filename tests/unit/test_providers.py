"""Wire-format tests for every provider (no network: httpx.MockTransport)."""

from __future__ import annotations

import json

import httpx
import pytest

from mao.config.schema import ModelConfig, ProviderConfig
from mao.core.errors import ContentFilterError
from mao.core.types import ChatMessage, CompletionRequest, FinishReason, MessageRole, ToolCall, ToolSpec
from mao.providers.anthropic import AnthropicProvider
from mao.providers.base import create_provider
from mao.providers.gemini import PLACEHOLDER_SIGNATURE, GeminiProvider, sanitize_schema
from mao.providers.ollama import OllamaProvider
from mao.providers.openai_chat import OpenAIChatProvider
from mao.providers.openai_responses import OpenAIResponsesProvider
from mao.providers.prompt_tools import parse_prompt_response, to_prompt_request
from mao.core.types import CompletionResponse

TOOL = ToolSpec(
    name="read_file",
    description="Read a file",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}, "limit": {"type": ["integer", "null"]}},
        "required": ["path"],
        "additionalProperties": False,
    },
)


def _provider(cls, type_name: str, handler=None, **options):  # type: ignore[no-untyped-def]
    config = ProviderConfig(type=type_name, options=options)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler or (lambda r: httpx.Response(500))))
    return cls("p", config, client)


def _history(model_id: str, provider_data: dict | None = None) -> CompletionRequest:
    call = ToolCall(id="call_1", name="read_file", arguments={"path": "a.py"})
    assistant = ChatMessage.assistant("I am reading the file.", [call])
    assistant.provider_data = provider_data
    return CompletionRequest(
        model=model_id,
        system="You are an agent.",
        messages=[
            ChatMessage.user("Analyse a.py"),
            assistant,
            ChatMessage.tool_result(call, "print('hi')"),
        ],
        tools=[TOOL],
        temperature=0.2,
        max_output_tokens=1000,
    )


# ------------------------------------------------------------------ OpenAI Responses


def test_openai_responses_body_plain_history() -> None:
    provider = _provider(OpenAIResponsesProvider, "openai_responses")
    body = provider.build_body(_history("gpt-x"), ModelConfig(reasoning=True, supports_temperature=False))
    assert body["store"] is False
    assert body["instructions"] == "You are an agent."
    assert body["include"] == ["reasoning.encrypted_content"]
    assert "temperature" not in body
    assert body["max_output_tokens"] == 1000
    assert body["tools"][0] == {"type": "function", "name": "read_file", "description": "Read a file", "parameters": TOOL.parameters}
    assert body["input"][0] == {"role": "user", "content": "Analyse a.py"}
    assert body["input"][2] == {"type": "function_call", "call_id": "call_1", "name": "read_file", "arguments": '{"path": "a.py"}'}
    assert body["input"][3] == {"type": "function_call_output", "call_id": "call_1", "output": "print('hi')"}


def test_openai_responses_native_replay_only_for_same_model() -> None:
    provider = _provider(OpenAIResponsesProvider, "openai_responses")
    native = {"format": "openai_responses", "provider": "p", "model": "gpt-x", "output": [{"type": "reasoning", "id": "rs_1", "encrypted_content": "enc"}]}
    same = provider.build_body(_history("gpt-x", native), ModelConfig())
    assert same["input"][1] == {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc"}
    other = provider.build_body(_history("gpt-y", native), ModelConfig())
    assert other["input"][1] == {"role": "assistant", "content": "I am reading the file."}


def test_openai_responses_parse() -> None:
    provider = _provider(OpenAIResponsesProvider, "openai_responses")
    data = {
        "id": "resp_1",
        "status": "completed",
        "model": "gpt-x",
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "reasoning", "id": "rs_2", "encrypted_content": "abc"},
            {"type": "message", "content": [{"type": "output_text", "text": "Hello"}]},
            {"type": "function_call", "id": "fc_1", "call_id": "call_9", "name": "read_file", "arguments": '{"path": "x"}'},
        ],
        "usage": {"input_tokens": 100, "output_tokens": 20, "input_tokens_details": {"cached_tokens": 40}, "output_tokens_details": {"reasoning_tokens": 5}},
    }
    response = provider.parse_response(data, CompletionRequest(model="gpt-x", messages=[]))
    assert response.message.content == "Hello"
    assert response.message.tool_calls[0].id == "call_9"
    assert response.message.tool_calls[0].arguments == {"path": "x"}
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.usage.cached_input_tokens == 40 and response.usage.reasoning_tokens == 5
    replay = response.message.provider_data["output"]
    assert [item.get("id") for item in replay] == ["rs_2", None, "fc_1"]
    incomplete = provider.parse_response({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []}, CompletionRequest(model="m", messages=[]))
    assert incomplete.finish_reason is FinishReason.LENGTH


async def test_openai_responses_http_roundtrip() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}], "usage": {"input_tokens": 3, "output_tokens": 1}})

    provider = _provider(OpenAIResponsesProvider, "openai_responses", handler)
    response = await provider.complete(CompletionRequest(model="gpt-x", messages=[ChatMessage.user("hi")]), api_key="sk-test", model=ModelConfig(), timeout=5)
    assert seen["url"] == "https://api.openai.com/v1/responses"
    assert seen["auth"] == "Bearer sk-test"
    assert response.message.content == "ok" and response.usage.input_tokens == 3


# ------------------------------------------------------------------ OpenAI-compatible chat


def test_openai_chat_body_and_parse() -> None:
    provider = _provider(OpenAIChatProvider, "openai_chat", max_tokens_param="max_completion_tokens")
    body = provider.build_body(_history("local-model"), ModelConfig())
    assert body["messages"][0] == {"role": "system", "content": "You are an agent."}
    assert body["messages"][2]["tool_calls"][0]["function"] == {"name": "read_file", "arguments": '{"path": "a.py"}'}
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "call_1", "content": "print('hi')"}
    assert body["max_completion_tokens"] == 1000 and body["temperature"] == 0.2
    data = {
        "choices": [{"message": {"content": "<think>hmm</think>Done", "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": "{bad"}}]}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0012},
    }
    response = provider.parse_response(data, CompletionRequest(model="m", messages=[]))
    assert response.message.content == "Done"
    assert response.message.tool_calls[0].parse_error
    assert response.usage.reported_cost_usd == 0.0012


# ------------------------------------------------------------------ Anthropic


def test_anthropic_messages_merge_tool_results_and_plain_history() -> None:
    provider = _provider(AnthropicProvider, "anthropic")
    request = _history("claude-x")
    call2 = ToolCall(id="call_2", name="read_file", arguments={"path": "b.py"})
    request.messages[1].tool_calls.append(call2)
    request.messages.append(ChatMessage.tool_result(call2, "boom", is_error=True))
    body = provider.build_body(request, ModelConfig(supports_temperature=False))
    assert body["system"] == "You are an agent."
    assert "temperature" not in body and body["max_tokens"] == 1000
    assert body["tools"][0]["input_schema"] == TOOL.parameters
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][1]["content"][1] == {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "a.py"}}
    results = body["messages"][2]["content"]
    assert [b["tool_use_id"] for b in results] == ["call_1", "call_2"]
    assert results[1]["is_error"] is True


def test_anthropic_replays_thinking_blocks_for_same_model() -> None:
    provider = _provider(AnthropicProvider, "anthropic")
    blocks = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "a.py"}},
    ]
    native = {"format": "anthropic", "provider": "p", "model": "claude-x", "content": blocks}
    body = provider.build_body(_history("claude-x", native), ModelConfig())
    assert body["messages"][1]["content"] == blocks
    body_other = provider.build_body(_history("claude-y", native), ModelConfig())
    assert all(b["type"] != "thinking" for b in body_other["messages"][1]["content"])
    plain = _history("claude-x", native)
    plain.metadata["plain_history"] = True
    assert all(b["type"] != "thinking" for b in provider.build_body(plain, ModelConfig())["messages"][1]["content"])


def test_anthropic_parse_usage_and_stop() -> None:
    provider = _provider(AnthropicProvider, "anthropic")
    data = {
        "id": "msg_1",
        "type": "message",
        "content": [{"type": "thinking", "thinking": "", "signature": "s"}, {"type": "text", "text": "Ok"}, {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "z"}}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 50, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 10, "output_tokens": 30},
    }
    response = provider.parse_response(data, CompletionRequest(model="claude-x", messages=[]))
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.usage.input_tokens == 160 and response.usage.cached_input_tokens == 100
    assert response.message.provider_data["content"][0]["type"] == "thinking"


async def test_anthropic_list_models_pagination() -> None:
    pages = [
        {"data": [{"id": "claude-a", "display_name": "A", "max_input_tokens": 1000000, "max_tokens": 128000, "capabilities": {"thinking": {"supported": True}}}], "has_more": True, "last_id": "claude-a"},
        {"data": [{"id": "claude-b", "max_input_tokens": 0, "max_tokens": 0}], "has_more": False, "last_id": "claude-b"},
    ]
    seen_headers: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers)
        return httpx.Response(200, json=pages[len(seen_headers) - 1])

    provider = _provider(AnthropicProvider, "anthropic", handler)
    models = await provider.list_models(api_key="sk-ant-test")
    assert [m.id for m in models] == ["claude-a", "claude-b"]
    assert models[0].context_window == 1000000 and "reasoning" in models[0].capabilities
    assert models[1].context_window is None
    assert seen_headers[0]["x-api-key"] == "sk-ant-test" and seen_headers[0]["anthropic-version"] == "2023-06-01"


# ------------------------------------------------------------------ Gemini


def test_gemini_schema_sanitizing() -> None:
    schema = sanitize_schema(TOOL.parameters)
    assert "additionalProperties" not in schema
    assert schema["properties"]["limit"] == {"type": "integer", "nullable": True}


def test_gemini_contents_and_signatures() -> None:
    provider = _provider(GeminiProvider, "gemini")
    request = _history("gemini-x")
    request.messages[1].tool_calls.append(ToolCall(id="gemini_call_abc", name="read_file", arguments={"path": "b"}))
    body = provider.build_body(request, ModelConfig())
    contents = body["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    calls = [p for p in contents[1]["parts"] if "functionCall" in p]
    assert calls[0]["thoughtSignature"] == PLACEHOLDER_SIGNATURE and "thoughtSignature" not in calls[1]
    assert calls[0]["functionCall"]["id"] == "call_1" and "id" not in calls[1]["functionCall"]
    assert contents[2]["parts"][0]["functionResponse"] == {"name": "read_file", "response": {"result": "print('hi')"}, "id": "call_1"}
    assert body["systemInstruction"] == {"parts": [{"text": "You are an agent."}]}
    assert body["generationConfig"] == {"maxOutputTokens": 1000, "temperature": 0.2}
    native_parts = [{"functionCall": {"name": "read_file", "args": {"path": "a.py"}}, "thoughtSignature": "REAL"}]
    native = {"format": "gemini", "provider": "p", "model": "gemini-x", "parts": native_parts}
    replay = provider.build_body(_history("gemini-x", native), ModelConfig())
    assert replay["contents"][1]["parts"] == native_parts


def test_gemini_parse_response() -> None:
    provider = _provider(GeminiProvider, "gemini")
    data = {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "thinking", "thought": True}, {"functionCall": {"name": "read_file", "args": {"path": "q"}}, "thoughtSignature": "S"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 70, "candidatesTokenCount": 10, "thoughtsTokenCount": 25},
    }
    response = provider.parse_response(data, CompletionRequest(model="gemini-x", messages=[]))
    assert response.message.content == ""
    call = response.message.tool_calls[0]
    assert call.id.startswith("gemini_call_") and call.arguments == {"path": "q"}
    assert response.usage.output_tokens == 35 and response.usage.reasoning_tokens == 25
    assert response.finish_reason is FinishReason.TOOL_CALLS
    with pytest.raises(ContentFilterError):
        provider.parse_response({"promptFeedback": {"blockReason": "SAFETY"}}, CompletionRequest(model="m", messages=[]))


# ------------------------------------------------------------------ Ollama


def test_ollama_body_and_parse() -> None:
    provider = _provider(OllamaProvider, "ollama", num_ctx=8192, keep_alive="5m")
    body = provider.build_body(_history("qwen3:4b"), ModelConfig(context_window=32000))
    assert body["options"] == {"num_ctx": 8192, "temperature": 0.2, "num_predict": 1000}
    assert body["keep_alive"] == "5m" and body["stream"] is False
    assert body["messages"][2]["tool_calls"][0]["function"]["arguments"] == {"path": "a.py"}
    assert body["messages"][3] == {"role": "tool", "content": "print('hi')", "tool_name": "read_file"}
    data = {"model": "qwen3:4b", "message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"index": 0, "name": "read_file", "arguments": {"path": "k"}}}]}, "done_reason": "stop", "prompt_eval_count": 12, "eval_count": 4}
    response = provider.parse_response(data, CompletionRequest(model="qwen3:4b", messages=[]))
    assert response.message.tool_calls[0].arguments == {"path": "k"}
    assert response.usage.input_tokens == 12 and response.finish_reason is FinishReason.TOOL_CALLS


async def test_ollama_discovery_marks_broken_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "good:latest", "size": 10}, {"name": "broken:latest", "size": 20}]})
        name = json.loads(request.content)["model"]
        if name == "broken:latest":
            return httpx.Response(404, json={"error": "model 'broken:latest' not found"})
        return httpx.Response(200, json={"capabilities": ["completion", "tools"], "model_info": {"qwen3.context_length": 40960}})

    provider = _provider(OllamaProvider, "ollama", handler)
    models = {m.id: m for m in await provider.list_models(api_key=None)}
    assert models["good:latest"].tool_calling == "native" and models["good:latest"].context_window == 40960
    assert "error" in models["broken:latest"].details


# ------------------------------------------------------------------ prompt-based tools


def test_prompt_tool_protocol_roundtrip() -> None:
    request = _history("tiny")
    converted = to_prompt_request(request)
    assert converted.tools == []
    assert "```tool" in converted.system and "read_file" in converted.system
    assert [m.role for m in converted.messages] == [MessageRole.USER, MessageRole.ASSISTANT, MessageRole.USER]
    assert '"name": "read_file"' in converted.messages[1].content
    raw = CompletionResponse(message=ChatMessage.assistant('I will have a look.\n```tool\n{"name": "read_file", "arguments": {"path": "src/app.py"}}\n```'))
    parsed = parse_prompt_response(raw, {"read_file"})
    assert parsed.finish_reason is FinishReason.TOOL_CALLS
    assert parsed.message.tool_calls[0].arguments == {"path": "src/app.py"}
    assert parsed.message.content == "I will have a look."


def test_create_provider_unknown_type() -> None:
    from mao.core.errors import ConfigError

    with pytest.raises(ConfigError):
        create_provider("x", ProviderConfig(type="nope"), httpx.AsyncClient())
