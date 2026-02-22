"""Tests for ServerlessResource."""

import pytest

from meganova.models.serverless import ServerlessModelsResponse
from meganova.resources.serverless import ServerlessResource
from tests.conftest import make_serverless_response


class TestServerlessListModels:
    def test_basic(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_serverless_response()
        resource = ServerlessResource(mock_sync_transport)

        result = resource.list_models()
        assert isinstance(result, ServerlessModelsResponse)
        assert result.count == 1

    def test_default_modality(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_serverless_response()
        resource = ServerlessResource(mock_sync_transport)

        resource.list_models()
        call_kwargs = mock_sync_transport.request.call_args.kwargs
        assert call_kwargs["params"]["modality"] == "text_generation"

    def test_custom_modality(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_serverless_response()
        resource = ServerlessResource(mock_sync_transport)

        resource.list_models(modality="text_to_image")
        call_kwargs = mock_sync_transport.request.call_args.kwargs
        assert call_kwargs["params"]["modality"] == "text_to_image"

    def test_uses_base_url_override(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_serverless_response()
        resource = ServerlessResource(mock_sync_transport)

        resource.list_models()
        call_kwargs = mock_sync_transport.request.call_args.kwargs
        assert call_kwargs["base_url_override"] == "https://api.meganova.ai"

    def test_correct_path(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_serverless_response()
        resource = ServerlessResource(mock_sync_transport)

        resource.list_models()
        args = mock_sync_transport.request.call_args
        assert args.args[1] == "/api/v1/serverless/models/filter"

    def test_model_data_parsed(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_serverless_response()
        resource = ServerlessResource(mock_sync_transport)

        result = resource.list_models()
        assert result.models[0].model_name == "llama-3"
