"""Tests for CloudAgent."""

from unittest.mock import MagicMock, patch

import pytest

from meganova.cloud.agent import CloudAgent, Conversation
from meganova.cloud.models import (
    AgentChatResponse,
    AgentInfo,
    ChatCompletionChunk,
    ChatCompletionResponse,
    PendingToolCall,
)
from meganova.errors import MeganovaError
from tests.conftest import (
    make_agent_chat_response,
    make_agent_info,
    make_cloud_completion_response,
)


class TestCloudAgentInit:
    def test_basic_init(self):
        agent = CloudAgent(api_key="agent_test123")
        assert agent._api_key == "agent_test123"

    def test_invalid_key_raises(self):
        with pytest.raises(MeganovaError, match="agent_"):
            CloudAgent(api_key="bad_key")

    def test_empty_key_raises(self):
        with pytest.raises(MeganovaError):
            CloudAgent(api_key="")

    def test_base_path_includes_key(self):
        agent = CloudAgent(api_key="agent_abc")
        assert agent._base_path == "/agents/v1/agent_abc"

    def test_custom_base_url(self):
        agent = CloudAgent(api_key="agent_abc", base_url="https://custom.api")
        assert agent._transport.base_url == "https://custom.api"


class TestCloudAgentInfo:
    def test_info_returns_agent_info(self):
        agent = CloudAgent(api_key="agent_test")
        agent._transport = MagicMock()
        agent._transport.request.return_value = make_agent_info()

        info = agent.info()
        assert isinstance(info, AgentInfo)
        assert info.name == "Test Agent"

    def test_info_calls_correct_path(self):
        agent = CloudAgent(api_key="agent_test")
        agent._transport = MagicMock()
        agent._transport.request.return_value = make_agent_info()

        agent.info()
        agent._transport.request.assert_called_once_with(
            "GET", "/agents/v1/agent_test/info"
        )


class TestCloudAgentChat:
    def test_basic_chat(self):
        agent = CloudAgent(api_key="agent_test")
        agent._transport = MagicMock()
        agent._transport.request.return_value = make_agent_chat_response()

        resp = agent.chat("Hello")
        assert isinstance(resp, AgentChatResponse)
        assert resp.response == "I can help with that."

    def test_chat_with_conversation_id(self):
        agent = CloudAgent(api_key="agent_test")
        agent._transport = MagicMock()
        agent._transport.request.return_value = make_agent_chat_response()

        agent.chat("Hi", conversation_id="conv_existing")
        body = agent._transport.request.call_args.kwargs["json_body"]
        assert body["conversation_id"] == "conv_existing"

    def test_chat_with_user_identifier(self):
        agent = CloudAgent(api_key="agent_test")
        agent._transport = MagicMock()
        agent._transport.request.return_value = make_agent_chat_response()

        agent.chat("Hi", user_identifier="user@example.com", user_identifier_type="email")
        body = agent._transport.request.call_args.kwargs["json_body"]
        assert body["user_identifier"] == "user@example.com"
        assert body["user_identifier_type"] == "email"

    def test_chat_with_extra_data(self):
        agent = CloudAgent(api_key="agent_test")
        agent._transport = MagicMock()
        agent._transport.request.return_value = make_agent_chat_response()

        agent.chat("Hi", extra_data={"custom": "value"})
        body = agent._transport.request.call_args.kwargs["json_body"]
        assert body["extra_data"]["custom"] == "value"

    def test_chat_correct_path(self):
        agent = CloudAgent(api_key="agent_test")
        agent._transport = MagicMock()
        agent._transport.request.return_value = make_agent_chat_response()

        agent.chat("Hi")
        args = agent._transport.request.call_args
        assert args.args[1] == "/agents/v1/agent_test/chat"

    def test_none_optionals_excluded(self):
        agent = CloudAgent(api_key="agent_test")
        agent._transport = MagicMock()
        agent._transport.request.return_value = make_agent_chat_response()

        agent.chat("Hi")
        body = agent._transport.request.call_args.kwargs["json_body"]
        assert "conversation_id" not in body
        assert "user_identifier" not in body


class TestCloudAgentConfirmTool:
    def test_approve(self):
        agent = CloudAgent(api_key="agent_test")
        agent._transport = MagicMock()
        agent._transport.request.return_value = make_agent_chat_response()

        resp = agent.confirm_tool("app_1", action="approve")
        assert isinstance(resp, AgentChatResponse)
        body = agent._transport.request.call_args.kwargs["json_body"]
        assert body["approval_id"] == "app_1"
        assert body["action"] == "approve"

    def test_reject(self):
        agent = CloudAgent(api_key="agent_test")
        agent._transport = MagicMock()
        agent._transport.request.return_value = make_agent_chat_response()

        agent.confirm_tool("app_1", action="reject")
        body = agent._transport.request.call_args.kwargs["json_body"]
        assert body["action"] == "reject"

    def test_invalid_action_raises(self):
        agent = CloudAgent(api_key="agent_test")
        with pytest.raises(MeganovaError, match="approve.*reject"):
            agent.confirm_tool("app_1", action="maybe")

    def test_correct_path(self):
        agent = CloudAgent(api_key="agent_test")
        agent._transport = MagicMock()
        agent._transport.request.return_value = make_agent_chat_response()

        agent.confirm_tool("app_1")
        args = agent._transport.request.call_args
        assert args.args[1] == "/agents/v1/agent_test/confirm-tool"


class TestCloudAgentCompletions:
    def test_non_streaming(self):
        agent = CloudAgent(api_key="agent_test")
        agent._transport = MagicMock()
        agent._transport.request.return_value = make_cloud_completion_response()

        resp = agent.completions([{"role": "user", "content": "Hi"}])
        assert isinstance(resp, ChatCompletionResponse)

    def test_streaming(self):
        agent = CloudAgent(api_key="agent_test")
        agent._transport = MagicMock()

        chunk_data = {
            "id": "c1", "object": "chat.completion.chunk",
            "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": {"content": "Hi"}}],
        }
        agent._transport.stream_sse.return_value = iter([chunk_data])

        result = agent.completions(
            [{"role": "user", "content": "Hi"}], stream=True,
        )
        chunks = list(result)
        assert len(chunks) == 1
        assert isinstance(chunks[0], ChatCompletionChunk)

    def test_completions_payload(self):
        agent = CloudAgent(api_key="agent_test")
        agent._transport = MagicMock()
        agent._transport.request.return_value = make_cloud_completion_response()

        agent.completions(
            [{"role": "user", "content": "Hi"}],
            temperature=0.5, max_tokens=100, user="user_1",
        )
        body = agent._transport.request.call_args.kwargs["json_body"]
        assert body["temperature"] == 0.5
        assert body["max_tokens"] == 100
        assert body["user"] == "user_1"

    def test_completions_correct_path(self):
        agent = CloudAgent(api_key="agent_test")
        agent._transport = MagicMock()
        agent._transport.request.return_value = make_cloud_completion_response()

        agent.completions([{"role": "user", "content": "Hi"}])
        args = agent._transport.request.call_args
        assert args.args[1] == "/agents/v1/agent_test/chat/completions"


class TestCloudAgentConversation:
    def test_creates_conversation(self):
        agent = CloudAgent(api_key="agent_test")
        conv = agent.conversation()
        assert isinstance(conv, Conversation)
        assert conv.conversation_id is None

    def test_creates_conversation_with_id(self):
        agent = CloudAgent(api_key="agent_test")
        conv = agent.conversation(conversation_id="conv_existing")
        assert conv.conversation_id == "conv_existing"
