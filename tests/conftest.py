"""Shared test fixtures for the meganova-python test suite."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Response data factories
# ---------------------------------------------------------------------------

def make_chat_response(
    content="Hello!",
    model="test-model",
    finish_reason="stop",
    tool_calls=None,
    usage=True,
):
    """Build a raw dict matching the ChatResponse schema."""
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = None
    resp = {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else finish_reason,
            }
        ],
    }
    if usage:
        resp["usage"] = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }
    return resp


def make_tool_call(name="get_weather", arguments='{"city":"NYC"}', call_id="call_abc"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def make_stream_chunk(content="Hi", finish_reason=None, model="test-model"):
    return {
        "id": "chatcmpl-stream1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }


def make_image_response(url="https://img.example.com/cat.png", n=1):
    return {
        "data": [{"url": url}] * n,
        "created": 1700000000,
    }


def make_embedding_response(dim=3):
    return {
        "data": [{"index": 0, "embedding": [0.1] * dim, "object": "embedding"}],
        "model": "text-embedding-ada-002",
        "object": "list",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


def make_video_generation(status="processing", vid_id="vid_123"):
    return {
        "id": vid_id,
        "object": "video.generation",
        "created_at": 1700000000,
        "status": status,
        "model": "veo-3",
    }


def make_model_info(model_id="gpt-4", **kwargs):
    base = {"id": model_id, "object": "model", "owned_by": "meganova"}
    base.update(kwargs)
    return base


def make_models_list(*model_ids):
    ids = model_ids or ("gpt-4", "claude-3")
    return {"data": [make_model_info(m) for m in ids]}


def make_serverless_response():
    return {
        "data": {
            "models": [
                {
                    "model_name": "llama-3",
                    "modality": "text_generation",
                    "cost_per_1k_input": 0.001,
                    "cost_per_1k_output": 0.002,
                }
            ],
            "count": 1,
        }
    }


def make_billing_response():
    return {
        "data": [
            {
                "vm_access_info_id": 1,
                "running_hours": 2.5,
                "fee_accumulated": 5.0,
                "tax_rate": 0.1,
                "fee_accumulated_taxed": 5.5,
                "product_status": "running",
            }
        ],
        "total_cost": 5.5,
        "message": "ok",
        "status": "success",
    }


def make_transcription_response(text="Hello world"):
    return {"text": text}


def make_agent_info():
    return {
        "id": "agent_123",
        "name": "Test Agent",
        "description": "A test agent",
        "welcome_message": "Hi there!",
        "is_available": True,
    }


def make_agent_chat_response(response_text="I can help with that."):
    return {
        "response": response_text,
        "conversation_id": "conv_abc",
        "message_id": "msg_001",
        "agent_id": "agent_123",
        "agent_name": "Test Agent",
        "tokens_used": 50,
        "memories_used": 0,
    }


def make_cloud_completion_response(content="Hello from cloud"):
    return {
        "id": "chatcmpl-cloud1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "agent-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


# ---------------------------------------------------------------------------
# Mock transports
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_sync_transport():
    transport = MagicMock()
    transport.base_url = "https://api.meganova.ai/v1"
    transport.api_key = "test-key"
    transport.timeout = 60.0
    transport.max_retries = 2
    transport.user_agent = "meganova-python/0.3.0"
    return transport


@pytest.fixture
def mock_async_transport():
    transport = AsyncMock()
    transport.base_url = "https://api.meganova.ai/v1"
    transport.api_key = "test-key"
    transport.timeout = 60.0
    transport.max_retries = 2
    transport.user_agent = "meganova-python/0.3.0"
    return transport


@pytest.fixture
def mock_cloud_transport():
    transport = MagicMock()
    transport.base_url = "https://studio-api.meganova.ai"
    transport.timeout = 60.0
    transport.max_retries = 2
    return transport


@pytest.fixture
def mock_client(mock_sync_transport):
    """MegaNova client with a mocked transport."""
    client = MagicMock()
    client._transport = mock_sync_transport
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = MagicMock()
    return client
