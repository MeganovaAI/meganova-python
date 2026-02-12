"""CloudAgent — chat with deployed MegaNova Cloud agents."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Union

from ..config import CLOUD_API_URL, DEFAULT_TIMEOUT, MAX_RETRIES
from ..errors import MeganovaError
from .models import (
    AgentChatResponse,
    AgentInfo,
    ChatCompletionChunk,
    ChatCompletionResponse,
    PendingToolCall,
)
from .transport import CloudTransport


class CloudAgent:
    """Client for a deployed MegaNova Cloud agent.

    Unlike the main ``MegaNova`` client, this uses the public agent API
    where the API key is part of the URL path — no Bearer token needed.

    Example::

        from meganova.cloud import CloudAgent

        agent = CloudAgent(api_key="agent_xxx...")
        info = agent.info()
        print(info.name, info.welcome_message)

        response = agent.chat("Hello!")
        print(response.response)
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = CLOUD_API_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ):
        if not api_key or not api_key.startswith("agent_"):
            raise MeganovaError(
                "Invalid agent API key. "
                "Keys start with 'agent_' — you can find yours in the Studio agent detail page."
            )
        self._api_key = api_key
        self._base_path = f"/agents/v1/{api_key}"
        self._transport = CloudTransport(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def info(self) -> AgentInfo:
        """Get public info about this agent (name, description, availability)."""
        data = self._transport.request("GET", f"{self._base_path}/info")
        return AgentInfo.model_validate(data)

    def chat(
        self,
        message: str,
        *,
        conversation_id: Optional[str] = None,
        user_identifier: Optional[str] = None,
        user_identifier_type: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> AgentChatResponse:
        """Send a message and get a response.

        Args:
            message: The user message.
            conversation_id: Continue an existing conversation. Omit to start new.
            user_identifier: Optional user ID for memory/personalization.
            user_identifier_type: Type of identifier (email, phone, user_id, anonymous).
            extra_data: Arbitrary metadata passed to the agent.

        Returns:
            AgentChatResponse with the agent's reply, conversation_id, and metadata.
        """
        body: Dict[str, Any] = {"message": message}
        if conversation_id is not None:
            body["conversation_id"] = conversation_id
        if user_identifier is not None:
            body["user_identifier"] = user_identifier
        if user_identifier_type is not None:
            body["user_identifier_type"] = user_identifier_type
        if extra_data is not None:
            body["extra_data"] = extra_data

        data = self._transport.request("POST", f"{self._base_path}/chat", json_body=body)
        return AgentChatResponse.model_validate(data)

    def confirm_tool(
        self,
        approval_id: str,
        action: str = "approve",
    ) -> AgentChatResponse:
        """Approve or reject a pending tool call.

        Args:
            approval_id: The approval_id from a PendingToolCall.
            action: "approve" or "reject".

        Returns:
            AgentChatResponse with the tool execution result (or rejection ack).
        """
        if action not in ("approve", "reject"):
            raise MeganovaError(f"action must be 'approve' or 'reject', got '{action}'")
        body = {"approval_id": approval_id, "action": action}
        data = self._transport.request("POST", f"{self._base_path}/confirm-tool", json_body=body)
        return AgentChatResponse.model_validate(data)

    def completions(
        self,
        messages: List[Dict[str, str]],
        *,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        user: Optional[str] = None,
    ) -> Union[ChatCompletionResponse, Iterator[ChatCompletionChunk]]:
        """OpenAI-compatible chat completions endpoint.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            stream: If True, returns an iterator of ChatCompletionChunk.
            temperature: Override the agent's configured temperature.
            max_tokens: Override the agent's configured max tokens.
            user: Maps to user_identifier for memory.

        Returns:
            ChatCompletionResponse (non-streaming) or Iterator[ChatCompletionChunk] (streaming).
        """
        body: Dict[str, Any] = {"messages": messages, "stream": stream}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if user is not None:
            body["user"] = user

        path = f"{self._base_path}/chat/completions"

        if stream:
            return self._stream_completions(path, body)

        data = self._transport.request("POST", path, json_body=body)
        return ChatCompletionResponse.model_validate(data)

    def _stream_completions(
        self, path: str, body: Dict[str, Any]
    ) -> Iterator[ChatCompletionChunk]:
        for chunk_data in self._transport.stream_sse(path, json_body=body):
            yield ChatCompletionChunk.model_validate(chunk_data)

    def conversation(self, conversation_id: Optional[str] = None) -> "Conversation":
        """Create a stateful conversation helper.

        Args:
            conversation_id: Resume an existing conversation, or None to start fresh.

        Returns:
            Conversation instance that auto-tracks conversation_id and pending tools.
        """
        return Conversation(self, conversation_id=conversation_id)


class Conversation:
    """Stateful wrapper around CloudAgent for multi-turn conversations.

    Automatically tracks conversation_id and pending tool calls.

    Example::

        conv = agent.conversation()
        r1 = conv.chat("I need to reset my password")
        r2 = conv.chat("My email is user@example.com")

        if conv.pending_tool_call:
            print(f"Agent wants to: {conv.pending_tool_call.description}")
            r3 = conv.confirm()  # or conv.reject()
    """

    def __init__(self, agent: CloudAgent, conversation_id: Optional[str] = None):
        self._agent = agent
        self.conversation_id = conversation_id
        self.messages: List[Dict[str, str]] = []
        self._pending_tool_call: Optional[PendingToolCall] = None

    @property
    def pending_tool_call(self) -> Optional[PendingToolCall]:
        """The current pending tool call, if any."""
        return self._pending_tool_call

    def chat(
        self,
        message: str,
        *,
        user_identifier: Optional[str] = None,
        user_identifier_type: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> AgentChatResponse:
        """Send a message, auto-continuing the conversation."""
        response = self._agent.chat(
            message,
            conversation_id=self.conversation_id,
            user_identifier=user_identifier,
            user_identifier_type=user_identifier_type,
            extra_data=extra_data,
        )
        self.conversation_id = response.conversation_id
        self.messages.append({"role": "user", "content": message})
        self.messages.append({"role": "assistant", "content": response.response})
        self._pending_tool_call = response.pending_tool_call
        return response

    def confirm(self) -> AgentChatResponse:
        """Approve the pending tool call."""
        if not self._pending_tool_call:
            raise MeganovaError("No pending tool call to confirm")
        response = self._agent.confirm_tool(self._pending_tool_call.approval_id, action="approve")
        self._pending_tool_call = response.pending_tool_call
        self.messages.append({"role": "assistant", "content": response.response})
        return response

    def reject(self) -> AgentChatResponse:
        """Reject the pending tool call."""
        if not self._pending_tool_call:
            raise MeganovaError("No pending tool call to reject")
        response = self._agent.confirm_tool(self._pending_tool_call.approval_id, action="reject")
        self._pending_tool_call = response.pending_tool_call
        self.messages.append({"role": "assistant", "content": response.response})
        return response
