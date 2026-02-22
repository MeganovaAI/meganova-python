"""Tests for cloud Pydantic models."""

import pytest

from meganova.cloud.models import (
    AgentChatResponse,
    AgentInfo,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionResponse,
    ChatCompletionUsage,
    PendingToolCall,
)


class TestAgentInfo:
    def test_basic(self):
        info = AgentInfo(
            id="agent_123", name="Bot", welcome_message="Hi!", is_available=True,
        )
        assert info.name == "Bot"
        assert info.is_available is True

    def test_optional_fields(self):
        info = AgentInfo(
            id="a", name="B", welcome_message="Hi", is_available=True,
            description="A bot", avatar_url="https://img.com/bot.png",
        )
        assert info.description == "A bot"

    def test_extra_ignored(self):
        info = AgentInfo(
            id="a", name="B", welcome_message="Hi", is_available=True,
            unknown_field="val",
        )
        assert not hasattr(info, "unknown_field")


class TestPendingToolCall:
    def test_basic(self):
        ptc = PendingToolCall(
            approval_id="app_1", tool_name="send_email",
            tool_arguments={"to": "user@example.com"},
            description="Send an email",
        )
        assert ptc.approval_id == "app_1"
        assert ptc.tool_name == "send_email"


class TestAgentChatResponse:
    def test_basic(self):
        resp = AgentChatResponse(
            response="Hello!", conversation_id="conv_1",
            message_id="msg_1", agent_id="a", agent_name="Bot",
        )
        assert resp.response == "Hello!"
        assert resp.conversation_id == "conv_1"
        assert resp.memories_used == 0

    def test_with_pending_tool(self):
        ptc = PendingToolCall(
            approval_id="app_1", tool_name="f",
            tool_arguments={}, description="d",
        )
        resp = AgentChatResponse(
            response="Need approval", conversation_id="c",
            message_id="m", agent_id="a", agent_name="B",
            pending_tool_call=ptc,
        )
        assert resp.pending_tool_call.approval_id == "app_1"

    def test_extra_ignored(self):
        resp = AgentChatResponse(
            response="Hi", conversation_id="c", message_id="m",
            agent_id="a", agent_name="B", extra="val",
        )
        assert not hasattr(resp, "extra")


class TestChatCompletionResponse:
    def test_basic(self):
        resp = ChatCompletionResponse(
            id="cc_1", created=1700000000, model="m",
            choices=[ChatCompletionChoice(
                message=ChatCompletionMessage(role="assistant", content="Hi"),
            )],
            usage=ChatCompletionUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        )
        assert resp.choices[0].message.content == "Hi"
        assert resp.usage.total_tokens == 15


class TestChatCompletionChunk:
    def test_basic(self):
        chunk = ChatCompletionChunk(
            id="cc_1", created=1700000000, model="m",
            choices=[ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(content="Hello"),
            )],
        )
        assert chunk.choices[0].delta.content == "Hello"
        assert chunk.object == "chat.completion.chunk"
