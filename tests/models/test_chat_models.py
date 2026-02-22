"""Tests for chat Pydantic models."""

import pytest

from meganova.models.chat import (
    ChatChoice,
    ChatMessage,
    ChatResponse,
    ChatStreamChunk,
    ChatStreamChunkChoice,
    FunctionCall,
    TokenUsage,
    ToolCall,
)


class TestFunctionCall:
    def test_basic(self):
        fc = FunctionCall(name="get_weather", arguments='{"city":"NYC"}')
        assert fc.name == "get_weather"
        assert fc.arguments == '{"city":"NYC"}'


class TestToolCall:
    def test_default_type(self):
        tc = ToolCall(id="call_1", function=FunctionCall(name="f", arguments="{}"))
        assert tc.type == "function"

    def test_custom_type(self):
        tc = ToolCall(id="call_1", type="custom", function=FunctionCall(name="f", arguments="{}"))
        assert tc.type == "custom"


class TestChatMessage:
    def test_user_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls is None

    def test_assistant_with_tool_calls(self):
        tc = ToolCall(id="call_1", function=FunctionCall(name="f", arguments="{}"))
        msg = ChatMessage(role="assistant", tool_calls=[tc])
        assert len(msg.tool_calls) == 1

    def test_tool_message(self):
        msg = ChatMessage(role="tool", content="result", tool_call_id="call_1")
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_1"


class TestTokenUsage:
    def test_basic(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.total_tokens == 30


class TestChatResponse:
    def test_minimal(self):
        resp = ChatResponse(
            id="chatcmpl-1",
            created=1700000000,
            model="gpt-4",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hi"),
                    finish_reason="stop",
                )
            ],
        )
        assert resp.model == "gpt-4"
        assert resp.choices[0].message.content == "Hi"
        assert resp.usage is None

    def test_with_usage(self):
        resp = ChatResponse(
            id="chatcmpl-1",
            created=1700000000,
            model="gpt-4",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hi"),
                )
            ],
            usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        )
        assert resp.usage.total_tokens == 15

    def test_default_object(self):
        resp = ChatResponse(
            id="c1", created=0, model="m",
            choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=""))],
        )
        assert resp.object == "chat.completion"

    def test_from_dict(self):
        data = {
            "id": "chatcmpl-1",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
        }
        resp = ChatResponse(**data)
        assert resp.choices[0].message.content == "Hi"


class TestChatStreamChunk:
    def test_basic(self):
        chunk = ChatStreamChunk(
            id="c1",
            created=1700000000,
            model="gpt-4",
            choices=[
                ChatStreamChunkChoice(index=0, delta={"content": "Hello"})
            ],
        )
        assert chunk.choices[0].delta["content"] == "Hello"
        assert chunk.object == "chat.completion.chunk"

    def test_finish_reason(self):
        chunk = ChatStreamChunk(
            id="c1", created=0, model="m",
            choices=[ChatStreamChunkChoice(index=0, delta={}, finish_reason="stop")],
        )
        assert chunk.choices[0].finish_reason == "stop"

    def test_from_dict(self):
        data = {
            "id": "c1", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": {"content": "Hi"}}],
        }
        chunk = ChatStreamChunk(**data)
        assert chunk.choices[0].delta["content"] == "Hi"
