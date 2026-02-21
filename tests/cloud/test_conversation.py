"""Tests for Conversation stateful wrapper."""

from unittest.mock import MagicMock

import pytest

from meganova.cloud.agent import CloudAgent, Conversation
from meganova.cloud.models import AgentChatResponse, PendingToolCall
from meganova.errors import MeganovaError


def _mock_agent():
    agent = MagicMock(spec=CloudAgent)
    agent._api_key = "agent_test"
    return agent


def _chat_response(text="Hello", conv_id="conv_1", pending=None):
    return AgentChatResponse(
        response=text,
        conversation_id=conv_id,
        message_id="msg_1",
        agent_id="agent_1",
        agent_name="Bot",
        pending_tool_call=pending,
    )


class TestConversationInit:
    def test_basic(self):
        conv = Conversation(_mock_agent())
        assert conv.conversation_id is None
        assert conv.messages == []
        assert conv.pending_tool_call is None

    def test_with_existing_id(self):
        conv = Conversation(_mock_agent(), conversation_id="conv_existing")
        assert conv.conversation_id == "conv_existing"


class TestConversationChat:
    def test_first_message(self):
        agent = _mock_agent()
        agent.chat.return_value = _chat_response("Hi there", "conv_new")

        conv = Conversation(agent)
        resp = conv.chat("Hello")

        assert resp.response == "Hi there"
        assert conv.conversation_id == "conv_new"
        assert len(conv.messages) == 2
        assert conv.messages[0] == {"role": "user", "content": "Hello"}
        assert conv.messages[1] == {"role": "assistant", "content": "Hi there"}

    def test_continues_conversation(self):
        agent = _mock_agent()
        agent.chat.return_value = _chat_response("R1", "conv_1")

        conv = Conversation(agent)
        conv.chat("M1")

        agent.chat.return_value = _chat_response("R2", "conv_1")
        conv.chat("M2")

        assert len(conv.messages) == 4
        # Second call should include conversation_id
        second_call = agent.chat.call_args_list[1]
        assert second_call.kwargs["conversation_id"] == "conv_1"

    def test_updates_conversation_id(self):
        agent = _mock_agent()
        agent.chat.return_value = _chat_response("R", "conv_new")

        conv = Conversation(agent)
        conv.chat("M")
        assert conv.conversation_id == "conv_new"

    def test_passes_user_identifier(self):
        agent = _mock_agent()
        agent.chat.return_value = _chat_response()

        conv = Conversation(agent)
        conv.chat("M", user_identifier="user@test.com", user_identifier_type="email")

        call_kwargs = agent.chat.call_args.kwargs
        assert call_kwargs["user_identifier"] == "user@test.com"
        assert call_kwargs["user_identifier_type"] == "email"

    def test_passes_extra_data(self):
        agent = _mock_agent()
        agent.chat.return_value = _chat_response()

        conv = Conversation(agent)
        conv.chat("M", extra_data={"key": "val"})

        call_kwargs = agent.chat.call_args.kwargs
        assert call_kwargs["extra_data"] == {"key": "val"}

    def test_tracks_pending_tool_call(self):
        ptc = PendingToolCall(
            approval_id="app_1", tool_name="send_email",
            tool_arguments={"to": "user@test.com"}, description="Send email",
        )
        agent = _mock_agent()
        agent.chat.return_value = _chat_response(pending=ptc)

        conv = Conversation(agent)
        conv.chat("Send an email")
        assert conv.pending_tool_call is not None
        assert conv.pending_tool_call.approval_id == "app_1"


class TestConversationConfirm:
    def test_confirm_pending(self):
        ptc = PendingToolCall(
            approval_id="app_1", tool_name="f",
            tool_arguments={}, description="d",
        )
        agent = _mock_agent()
        agent.chat.return_value = _chat_response(pending=ptc)
        agent.confirm_tool.return_value = _chat_response("Confirmed!")

        conv = Conversation(agent)
        conv.chat("Do something")

        resp = conv.confirm()
        assert resp.response == "Confirmed!"
        agent.confirm_tool.assert_called_once_with("app_1", action="approve")

    def test_confirm_without_pending_raises(self):
        conv = Conversation(_mock_agent())
        with pytest.raises(MeganovaError, match="No pending tool call"):
            conv.confirm()

    def test_confirm_clears_pending(self):
        ptc = PendingToolCall(
            approval_id="app_1", tool_name="f",
            tool_arguments={}, description="d",
        )
        agent = _mock_agent()
        agent.chat.return_value = _chat_response(pending=ptc)
        agent.confirm_tool.return_value = _chat_response("Done")

        conv = Conversation(agent)
        conv.chat("task")
        conv.confirm()
        assert conv.pending_tool_call is None

    def test_confirm_adds_to_messages(self):
        ptc = PendingToolCall(
            approval_id="app_1", tool_name="f",
            tool_arguments={}, description="d",
        )
        agent = _mock_agent()
        agent.chat.return_value = _chat_response(pending=ptc)
        agent.confirm_tool.return_value = _chat_response("Confirmed!")

        conv = Conversation(agent)
        conv.chat("task")
        conv.confirm()
        assert conv.messages[-1] == {"role": "assistant", "content": "Confirmed!"}


class TestConversationReject:
    def test_reject_pending(self):
        ptc = PendingToolCall(
            approval_id="app_1", tool_name="f",
            tool_arguments={}, description="d",
        )
        agent = _mock_agent()
        agent.chat.return_value = _chat_response(pending=ptc)
        agent.confirm_tool.return_value = _chat_response("Rejected")

        conv = Conversation(agent)
        conv.chat("task")

        resp = conv.reject()
        assert resp.response == "Rejected"
        agent.confirm_tool.assert_called_once_with("app_1", action="reject")

    def test_reject_without_pending_raises(self):
        conv = Conversation(_mock_agent())
        with pytest.raises(MeganovaError, match="No pending tool call"):
            conv.reject()

    def test_reject_adds_to_messages(self):
        ptc = PendingToolCall(
            approval_id="app_1", tool_name="f",
            tool_arguments={}, description="d",
        )
        agent = _mock_agent()
        agent.chat.return_value = _chat_response(pending=ptc)
        agent.confirm_tool.return_value = _chat_response("Rejected")

        conv = Conversation(agent)
        conv.chat("task")
        conv.reject()
        assert conv.messages[-1] == {"role": "assistant", "content": "Rejected"}
