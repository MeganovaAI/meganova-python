"""Tests for the sync MegaNova client."""

import pytest

from meganova.client import MegaNova
from meganova.config import PRODUCTION_API_URL, DEFAULT_TIMEOUT, MAX_RETRIES
from meganova.resources.chat import Chat
from meganova.resources.models_ import ModelsResource
from meganova.resources.images import ImagesResource
from meganova.resources.audio import AudioResource
from meganova.resources.embeddings import EmbeddingsResource
from meganova.resources.videos import VideosResource
from meganova.resources.serverless import ServerlessResource
from meganova.resources.billing import BillingResource
from meganova.resources.usage import UsageResource


class TestMegaNovaInit:
    def test_basic_init(self):
        client = MegaNova(api_key="test-key-123")
        assert client._transport.api_key == "test-key-123"

    def test_empty_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key is required"):
            MegaNova(api_key="")

    def test_default_base_url(self):
        client = MegaNova(api_key="k")
        assert client._transport.base_url == PRODUCTION_API_URL

    def test_custom_base_url(self):
        client = MegaNova(api_key="k", base_url="https://custom.api/v1")
        assert client._transport.base_url == "https://custom.api/v1"

    def test_trailing_slash_stripped(self):
        client = MegaNova(api_key="k", base_url="https://custom.api/v1/")
        assert not client._transport.base_url.endswith("/")

    def test_default_timeout(self):
        client = MegaNova(api_key="k")
        assert client._transport.timeout == DEFAULT_TIMEOUT

    def test_custom_timeout(self):
        client = MegaNova(api_key="k", timeout=120.0)
        assert client._transport.timeout == 120.0

    def test_default_max_retries(self):
        client = MegaNova(api_key="k")
        assert client._transport.max_retries == MAX_RETRIES

    def test_custom_max_retries(self):
        client = MegaNova(api_key="k", max_retries=5)
        assert client._transport.max_retries == 5

    def test_user_agent_contains_version(self):
        client = MegaNova(api_key="k")
        assert "meganova-python/" in client._transport.user_agent

    def test_user_agent_extra(self):
        client = MegaNova(api_key="k", user_agent_extra="myapp/1.0")
        assert "myapp/1.0" in client._transport.user_agent


class TestMegaNovaResources:
    def test_chat_resource(self):
        client = MegaNova(api_key="k")
        assert isinstance(client.chat, Chat)

    def test_models_resource(self):
        client = MegaNova(api_key="k")
        assert isinstance(client.models, ModelsResource)

    def test_images_resource(self):
        client = MegaNova(api_key="k")
        assert isinstance(client.images, ImagesResource)

    def test_audio_resource(self):
        client = MegaNova(api_key="k")
        assert isinstance(client.audio, AudioResource)

    def test_embeddings_resource(self):
        client = MegaNova(api_key="k")
        assert isinstance(client.embeddings, EmbeddingsResource)

    def test_videos_resource(self):
        client = MegaNova(api_key="k")
        assert isinstance(client.videos, VideosResource)

    def test_serverless_resource(self):
        client = MegaNova(api_key="k")
        assert isinstance(client.serverless, ServerlessResource)

    def test_billing_resource(self):
        client = MegaNova(api_key="k")
        assert isinstance(client.billing, BillingResource)

    def test_usage_resource(self):
        client = MegaNova(api_key="k")
        assert isinstance(client.usage, UsageResource)
