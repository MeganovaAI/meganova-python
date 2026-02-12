"""Pydantic models for the MegaNova Studio Agent API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class AgentInfo(BaseModel):
    """Public agent info returned by the /info endpoint."""
    id: str
    name: str
    description: Optional[str] = None
    avatar_url: Optional[str] = None
    welcome_message: str
    is_available: bool

    model_config = {"extra": "ignore"}


class PendingToolCall(BaseModel):
    """A tool call awaiting user confirmation."""
    approval_id: str
    tool_name: str
    tool_arguments: Dict[str, Any]
    description: str

    model_config = {"extra": "ignore"}


class AgentChatResponse(BaseModel):
    """Response from the /chat endpoint."""
    response: str
    conversation_id: str
    message_id: str
    agent_id: str
    agent_name: str
    tokens_used: Optional[int] = None
    memories_used: int = 0
    image_urls: Optional[List[str]] = None
    resolution_status: Optional[str] = None
    resolution_confidence: Optional[float] = None
    pending_tool_call: Optional[PendingToolCall] = None
    status: Optional[str] = None

    model_config = {"extra": "ignore"}


class ChatCompletionMessage(BaseModel):
    """Message within a chat completion response."""
    role: str
    content: Optional[str] = None

    model_config = {"extra": "ignore"}


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Optional[str] = "stop"

    model_config = {"extra": "ignore"}


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    model_config = {"extra": "ignore"}


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

    model_config = {"extra": "ignore"}


class ChatCompletionChunkDelta(BaseModel):
    """Delta within a streaming chunk choice."""
    role: Optional[str] = None
    content: Optional[str] = None

    model_config = {"extra": "ignore"}


class ChatCompletionChunkChoice(BaseModel):
    """A single choice in a streaming chunk."""
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None

    model_config = {"extra": "ignore"}


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]

    model_config = {"extra": "ignore"}
