"""Tests for web fetch tool."""

from unittest.mock import MagicMock, patch

import pytest

from meganova.agents.tools.web import web_fetch_tool, _web_fetch
from meganova.agents.tools.base import ToolDefinition


class TestWebFetchTool:
    def test_is_tool_definition(self):
        assert isinstance(web_fetch_tool, ToolDefinition)
        assert web_fetch_tool.name == "web_fetch"

    def test_has_parameters(self):
        assert "url" in web_fetch_tool.parameters["properties"]
        assert "url" in web_fetch_tool.parameters["required"]

    @patch("meganova.agents.tools.web.requests.request")
    def test_successful_fetch(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "Hello World"
        mock_request.return_value = mock_resp

        result = _web_fetch("https://example.com")
        assert "Status: 200" in result
        assert "Hello World" in result

    @patch("meganova.agents.tools.web.requests.request")
    def test_truncates_long_response(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "x" * 10000
        mock_request.return_value = mock_resp

        result = _web_fetch("https://example.com", max_length=100)
        assert "truncated" in result
        assert "10000 total chars" in result

    @patch("meganova.agents.tools.web.requests.request")
    def test_error_handling(self, mock_request):
        mock_request.side_effect = Exception("Connection refused")
        result = _web_fetch("https://bad.example.com")
        assert "Error fetching" in result

    @patch("meganova.agents.tools.web.requests.request")
    def test_custom_method(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "{}"
        mock_request.return_value = mock_resp

        _web_fetch("https://api.example.com", method="POST")
        mock_request.assert_called_once_with("POST", "https://api.example.com", timeout=15)

    @patch("meganova.agents.tools.web.requests.request")
    def test_default_get_method(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "ok"
        mock_request.return_value = mock_resp

        _web_fetch("https://example.com")
        mock_request.assert_called_once_with("GET", "https://example.com", timeout=15)
