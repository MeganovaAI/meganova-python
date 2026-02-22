"""Tests for ModelsResource."""

import pytest

from meganova.models.models_ import ModelInfo
from meganova.resources.models_ import ModelsResource
from tests.conftest import make_model_info, make_models_list


class TestModelsList:
    def test_list_returns_model_infos(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_models_list("gpt-4", "claude-3")
        resource = ModelsResource(mock_sync_transport)

        result = resource.list()
        assert len(result) == 2
        assert all(isinstance(m, ModelInfo) for m in result)

    def test_list_calls_correct_endpoint(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_models_list()
        resource = ModelsResource(mock_sync_transport)

        resource.list()
        mock_sync_transport.request.assert_called_once_with("GET", "/models")

    def test_filter_by_capability(self, mock_sync_transport):
        mock_sync_transport.request.return_value = {
            "data": [
                make_model_info("gpt-4", capabilities={"chat": True, "vision": True}),
                make_model_info("text-model", capabilities={"chat": True, "vision": False}),
            ]
        }
        resource = ModelsResource(mock_sync_transport)

        result = resource.list(capability="vision")
        assert len(result) == 1
        assert result[0].id == "gpt-4"

    def test_filter_by_type(self, mock_sync_transport):
        mock_sync_transport.request.return_value = {
            "data": [
                make_model_info("gpt-4", tags=["chat", "premium"]),
                make_model_info("whisper", tags=["audio"]),
            ]
        }
        resource = ModelsResource(mock_sync_transport)

        result = resource.list(type="audio")
        assert len(result) == 1
        assert result[0].id == "whisper"

    def test_no_capabilities_excluded_from_filter(self, mock_sync_transport):
        mock_sync_transport.request.return_value = {
            "data": [
                make_model_info("no-caps"),
                make_model_info("has-caps", capabilities={"chat": True}),
            ]
        }
        resource = ModelsResource(mock_sync_transport)

        result = resource.list(capability="chat")
        assert len(result) == 1
        assert result[0].id == "has-caps"

    def test_empty_list(self, mock_sync_transport):
        mock_sync_transport.request.return_value = {"data": []}
        resource = ModelsResource(mock_sync_transport)

        result = resource.list()
        assert result == []


class TestModelsGet:
    def test_get_by_id(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_model_info("gpt-4", name="GPT-4")
        resource = ModelsResource(mock_sync_transport)

        result = resource.get("gpt-4")
        assert isinstance(result, ModelInfo)
        assert result.name == "GPT-4"

    def test_get_calls_correct_path(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_model_info("gpt-4")
        resource = ModelsResource(mock_sync_transport)

        resource.get("gpt-4")
        mock_sync_transport.request.assert_called_once_with("GET", "/models/gpt-4")

    def test_get_with_slash_in_id(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_model_info("org/model")
        resource = ModelsResource(mock_sync_transport)

        resource.get("org/model")
        mock_sync_transport.request.assert_called_once_with("GET", "/models/org/model")
