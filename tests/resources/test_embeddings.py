"""Tests for EmbeddingsResource."""

import pytest

from meganova.models.embeddings import EmbeddingResponse
from meganova.resources.embeddings import EmbeddingsResource
from tests.conftest import make_embedding_response


class TestEmbeddingsCreate:
    def test_basic_embedding(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_embedding_response()
        resource = EmbeddingsResource(mock_sync_transport)

        result = resource.create(input="Hello", model="ada-002")
        assert isinstance(result, EmbeddingResponse)
        assert len(result.data) == 1

    def test_list_input(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_embedding_response()
        resource = EmbeddingsResource(mock_sync_transport)

        resource.create(input=["Hello", "World"], model="ada-002")
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["input"] == ["Hello", "World"]

    def test_correct_endpoint(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_embedding_response()
        resource = EmbeddingsResource(mock_sync_transport)

        resource.create(input="Hi", model="ada-002")
        args = mock_sync_transport.request.call_args
        assert args.args == ("POST", "/embeddings")

    def test_encoding_format_included(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_embedding_response()
        resource = EmbeddingsResource(mock_sync_transport)

        resource.create(input="Hi", model="ada-002", encoding_format="float")
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["encoding_format"] == "float"

    def test_encoding_format_excluded_when_none(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_embedding_response()
        resource = EmbeddingsResource(mock_sync_transport)

        resource.create(input="Hi", model="ada-002")
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert "encoding_format" not in payload

    def test_model_in_response(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_embedding_response()
        resource = EmbeddingsResource(mock_sync_transport)

        result = resource.create(input="Hi", model="ada-002")
        assert result.model == "text-embedding-ada-002"
