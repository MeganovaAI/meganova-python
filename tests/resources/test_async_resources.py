"""Tests for async resource variants."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from meganova.models.chat import ChatResponse, ChatStreamChunk
from meganova.models.embeddings import EmbeddingResponse
from meganova.models.images import ImageGenerationResponse
from meganova.models.models_ import ModelInfo
from meganova.models.videos import VideoGeneration
from meganova.resources.async_resources import (
    AsyncChat,
    AsyncCompletions,
    AsyncEmbeddingsResource,
    AsyncImagesResource,
    AsyncModelsResource,
    AsyncVideosResource,
)
from tests.conftest import (
    make_chat_response,
    make_embedding_response,
    make_image_response,
    make_model_info,
    make_models_list,
    make_video_generation,
)


class TestAsyncCompletions:
    async def test_basic_completion(self, mock_async_transport):
        mock_async_transport.request.return_value = make_chat_response(content="Hello async")
        comp = AsyncCompletions(mock_async_transport)

        result = await comp.create(messages=[{"role": "user", "content": "Hi"}], model="m")
        assert isinstance(result, ChatResponse)
        assert result.choices[0].message.content == "Hello async"

    async def test_payload_construction(self, mock_async_transport):
        mock_async_transport.request.return_value = make_chat_response()
        comp = AsyncCompletions(mock_async_transport)

        await comp.create(messages=[], model="gpt-4", temperature=0.5, max_tokens=100)
        payload = mock_async_transport.request.call_args.kwargs["json"]
        assert payload["model"] == "gpt-4"
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 100

    async def test_stream_returns_async_iterator(self, mock_async_transport):
        chunk_data = {
            "id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
        }

        async def mock_stream_sse(*args, **kwargs):
            yield chunk_data

        mock_async_transport.stream_sse = mock_stream_sse
        comp = AsyncCompletions(mock_async_transport)

        chunks = []
        result = await comp.create(messages=[], model="m", stream=True)
        async for c in result:
            chunks.append(c)
        assert len(chunks) == 1
        assert isinstance(chunks[0], ChatStreamChunk)


class TestAsyncChat:
    async def test_has_completions(self, mock_async_transport):
        chat = AsyncChat(mock_async_transport)
        assert isinstance(chat.completions, AsyncCompletions)


class TestAsyncEmbeddingsResource:
    async def test_basic(self, mock_async_transport):
        mock_async_transport.request.return_value = make_embedding_response()
        resource = AsyncEmbeddingsResource(mock_async_transport)

        result = await resource.create(input="Hello", model="ada-002")
        assert isinstance(result, EmbeddingResponse)

    async def test_encoding_format(self, mock_async_transport):
        mock_async_transport.request.return_value = make_embedding_response()
        resource = AsyncEmbeddingsResource(mock_async_transport)

        await resource.create(input="Hi", model="m", encoding_format="float")
        payload = mock_async_transport.request.call_args.kwargs["json"]
        assert payload["encoding_format"] == "float"


class TestAsyncImagesResource:
    async def test_basic(self, mock_async_transport):
        mock_async_transport.request.return_value = make_image_response()
        resource = AsyncImagesResource(mock_async_transport)

        result = await resource.generate("a cat", model="flux-1")
        assert isinstance(result, ImageGenerationResponse)

    async def test_size_from_wh(self, mock_async_transport):
        mock_async_transport.request.return_value = make_image_response()
        resource = AsyncImagesResource(mock_async_transport)

        await resource.generate("a cat", model="m", width=512, height=768)
        payload = mock_async_transport.request.call_args.kwargs["json"]
        assert payload["size"] == "512x768"


class TestAsyncVideosResource:
    async def test_basic(self, mock_async_transport):
        mock_async_transport.request.return_value = make_video_generation()
        resource = AsyncVideosResource(mock_async_transport)

        result = await resource.generate(prompt="a cat", model="veo-3")
        assert isinstance(result, VideoGeneration)

    async def test_get(self, mock_async_transport):
        mock_async_transport.request.return_value = make_video_generation(status="completed")
        resource = AsyncVideosResource(mock_async_transport)

        result = await resource.get("vid_123")
        assert result.status == "completed"


class TestAsyncModelsResource:
    async def test_list(self, mock_async_transport):
        mock_async_transport.request.return_value = make_models_list("gpt-4", "claude-3")
        resource = AsyncModelsResource(mock_async_transport)

        result = await resource.list()
        assert len(result) == 2

    async def test_filter_by_capability(self, mock_async_transport):
        mock_async_transport.request.return_value = {
            "data": [
                make_model_info("gpt-4", capabilities={"vision": True}),
                make_model_info("text-model", capabilities={"vision": False}),
            ]
        }
        resource = AsyncModelsResource(mock_async_transport)

        result = await resource.list(capability="vision")
        assert len(result) == 1

    async def test_get(self, mock_async_transport):
        mock_async_transport.request.return_value = make_model_info("gpt-4")
        resource = AsyncModelsResource(mock_async_transport)

        result = await resource.get("gpt-4")
        assert isinstance(result, ModelInfo)
