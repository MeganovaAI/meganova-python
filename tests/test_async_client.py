"""Tests for the async MegaNova client."""

import pytest

from meganova.async_client import AsyncMegaNova
from meganova.config import PRODUCTION_API_URL
from meganova.resources.async_resources import (
    AsyncChat,
    AsyncEmbeddingsResource,
    AsyncImagesResource,
    AsyncVideosResource,
    AsyncModelsResource,
    AsyncAudioResource,
)


class TestAsyncMegaNovaInit:
    def test_basic_init(self):
        client = AsyncMegaNova(api_key="test-key")
        assert client._transport.api_key == "test-key"

    def test_empty_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key is required"):
            AsyncMegaNova(api_key="")

    def test_default_base_url(self):
        client = AsyncMegaNova(api_key="k")
        assert client._transport.base_url == PRODUCTION_API_URL

    def test_custom_base_url(self):
        client = AsyncMegaNova(api_key="k", base_url="https://custom.api/v1")
        assert client._transport.base_url == "https://custom.api/v1"

    def test_user_agent_extra(self):
        client = AsyncMegaNova(api_key="k", user_agent_extra="myapp/2.0")
        assert "myapp/2.0" in client._transport.user_agent


class TestAsyncMegaNovaResources:
    def test_chat_resource(self):
        client = AsyncMegaNova(api_key="k")
        assert isinstance(client.chat, AsyncChat)

    def test_embeddings_resource(self):
        client = AsyncMegaNova(api_key="k")
        assert isinstance(client.embeddings, AsyncEmbeddingsResource)

    def test_images_resource(self):
        client = AsyncMegaNova(api_key="k")
        assert isinstance(client.images, AsyncImagesResource)

    def test_videos_resource(self):
        client = AsyncMegaNova(api_key="k")
        assert isinstance(client.videos, AsyncVideosResource)

    def test_models_resource(self):
        client = AsyncMegaNova(api_key="k")
        assert isinstance(client.models, AsyncModelsResource)

    def test_audio_resource(self):
        client = AsyncMegaNova(api_key="k")
        assert isinstance(client.audio, AsyncAudioResource)


class TestAsyncContextManager:
    async def test_aenter_returns_self(self):
        client = AsyncMegaNova(api_key="k")
        result = await client.__aenter__()
        assert result is client
        await client.close()
