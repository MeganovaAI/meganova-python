"""Tests for CloudTransport."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from meganova.cloud.transport import CloudTransport
from meganova.errors import APIError, AuthenticationError, MeganovaError, RateLimitError


def _make_transport(**kwargs):
    defaults = {
        "base_url": "https://studio-api.meganova.ai",
        "timeout": 60.0,
        "max_retries": 2,
    }
    defaults.update(kwargs)
    return CloudTransport(**defaults)


class TestCloudTransportInit:
    def test_base_url_stripped(self):
        t = _make_transport(base_url="https://studio-api.meganova.ai/")
        assert t.base_url == "https://studio-api.meganova.ai"


class TestCloudTransportRequest:
    def test_successful_response(self):
        t = _make_transport()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": "ok"}
        t._session.request = MagicMock(return_value=mock_resp)

        result = t.request("GET", "/agents/v1/key/info")
        assert result == {"result": "ok"}

    def test_no_authorization_header(self):
        t = _make_transport()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        t._session.request = MagicMock(return_value=mock_resp)

        t.request("GET", "/info")
        headers = t._session.request.call_args.kwargs["headers"]
        assert "Authorization" not in headers

    def test_user_agent_header(self):
        t = _make_transport()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        t._session.request = MagicMock(return_value=mock_resp)

        t.request("GET", "/info")
        headers = t._session.request.call_args.kwargs["headers"]
        assert "meganova-python" in headers["User-Agent"]

    def test_content_type_for_json(self):
        t = _make_transport()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        t._session.request = MagicMock(return_value=mock_resp)

        t.request("POST", "/chat", json_body={"message": "Hi"})
        headers = t._session.request.call_args.kwargs["headers"]
        assert headers["Content-Type"] == "application/json"

    def test_stream_returns_response(self):
        t = _make_transport()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        t._session.request = MagicMock(return_value=mock_resp)

        result = t.request("POST", "/completions", stream=True)
        assert result is mock_resp


class TestCloudTransportErrors:
    def test_401_raises_auth_error(self):
        t = _make_transport(max_retries=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"error": {"message": "invalid key"}}
        mock_resp.reason = "Unauthorized"
        t._session.request = MagicMock(return_value=mock_resp)

        with pytest.raises(AuthenticationError):
            t.request("GET", "/info")

    def test_429_raises_rate_limit(self):
        t = _make_transport(max_retries=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = {"error": {"message": "rate limited"}}
        mock_resp.reason = "Too Many Requests"
        t._session.request = MagicMock(return_value=mock_resp)

        with pytest.raises(RateLimitError):
            t.request("GET", "/info")

    def test_500_raises_api_error(self):
        t = _make_transport(max_retries=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": {"message": "internal error"}}
        mock_resp.reason = "Internal Server Error"
        t._session.request = MagicMock(return_value=mock_resp)

        with pytest.raises(APIError):
            t.request("GET", "/info")

    def test_detail_error_format(self):
        t = _make_transport(max_retries=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"detail": "bad request details"}
        mock_resp.reason = "Bad Request"
        t._session.request = MagicMock(return_value=mock_resp)

        with pytest.raises(APIError, match="bad request details"):
            t.request("POST", "/chat")


class TestCloudTransportRetry:
    @patch("meganova.cloud.transport.time.sleep")
    def test_retries_on_network_error(self, mock_sleep):
        t = _make_transport(max_retries=1)
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"ok": True}

        t._session.request = MagicMock(
            side_effect=[requests.ConnectionError("refused"), ok_resp]
        )

        result = t.request("GET", "/info")
        assert result == {"ok": True}

    @patch("meganova.cloud.transport.time.sleep")
    def test_exhausted_retries(self, mock_sleep):
        t = _make_transport(max_retries=1)
        t._session.request = MagicMock(
            side_effect=[
                requests.ConnectionError("fail1"),
                requests.ConnectionError("fail2"),
            ]
        )

        with pytest.raises(MeganovaError, match="Network error"):
            t.request("GET", "/info")


class TestCloudTransportSSE:
    def test_stream_sse_yields_chunks(self):
        t = _make_transport()
        chunk1 = {"id": "c1", "choices": [{"delta": {"content": "Hi"}}]}
        chunk2 = {"id": "c2", "choices": [{"delta": {"content": " there"}}]}

        lines = [
            f"data: {json.dumps(chunk1)}",
            f"data: {json.dumps(chunk2)}",
            "data: [DONE]",
        ]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)
        mock_resp.close = MagicMock()
        t._session.request = MagicMock(return_value=mock_resp)

        chunks = list(t.stream_sse("/completions", json_body={"stream": True}))
        assert len(chunks) == 2
        assert chunks[0]["id"] == "c1"
        mock_resp.close.assert_called_once()

    def test_stream_sse_skips_empty_and_invalid(self):
        t = _make_transport()
        chunk = {"id": "c1"}
        lines = [
            "",
            "data: {invalid}",
            f"data: {json.dumps(chunk)}",
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)
        mock_resp.close = MagicMock()
        t._session.request = MagicMock(return_value=mock_resp)

        chunks = list(t.stream_sse("/completions"))
        assert len(chunks) == 1
