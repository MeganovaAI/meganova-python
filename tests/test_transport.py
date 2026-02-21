"""Tests for SyncTransport."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from meganova.errors import APIError, AuthenticationError, MeganovaError, RateLimitError
from meganova.transport import SyncTransport


def _make_transport(**kwargs):
    defaults = {
        "base_url": "https://api.meganova.ai/v1",
        "api_key": "test-key",
        "timeout": 60.0,
        "max_retries": 2,
        "user_agent": "meganova-python/0.3.0",
    }
    defaults.update(kwargs)
    return SyncTransport(**defaults)


class TestSyncTransportInit:
    def test_base_url_trailing_slash_stripped(self):
        t = _make_transport(base_url="https://api.meganova.ai/v1/")
        assert t.base_url == "https://api.meganova.ai/v1"

    def test_stores_api_key(self):
        t = _make_transport()
        assert t.api_key == "test-key"

    def test_stores_timeout(self):
        t = _make_transport(timeout=30.0)
        assert t.timeout == 30.0


class TestSyncTransportRequest:
    def test_successful_json_response(self):
        t = _make_transport()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": "ok"}
        t._session.request = MagicMock(return_value=mock_resp)

        result = t.request("GET", "/models")
        assert result == {"result": "ok"}

    def test_stream_returns_response_object(self):
        t = _make_transport()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        t._session.request = MagicMock(return_value=mock_resp)

        result = t.request("POST", "/chat/completions", json={"a": 1}, stream=True)
        assert result is mock_resp

    def test_posts_json_payload(self):
        t = _make_transport()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        t._session.request = MagicMock(return_value=mock_resp)

        t.request("POST", "/chat/completions", json={"model": "gpt-4"})
        call_kwargs = t._session.request.call_args
        assert call_kwargs.kwargs["json"] == {"model": "gpt-4"}

    def test_sends_authorization_header(self):
        t = _make_transport()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        t._session.request = MagicMock(return_value=mock_resp)

        t.request("GET", "/models")
        call_kwargs = t._session.request.call_args
        headers = call_kwargs.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-key"

    def test_sends_user_agent_header(self):
        t = _make_transport()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        t._session.request = MagicMock(return_value=mock_resp)

        t.request("GET", "/models")
        headers = t._session.request.call_args.kwargs["headers"]
        assert headers["User-Agent"] == "meganova-python/0.3.0"

    def test_sends_sdk_header(self):
        t = _make_transport()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        t._session.request = MagicMock(return_value=mock_resp)

        t.request("GET", "/models")
        headers = t._session.request.call_args.kwargs["headers"]
        assert headers["X-MN-SDK"] == "python"

    def test_base_url_override(self):
        t = _make_transport()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        t._session.request = MagicMock(return_value=mock_resp)

        t.request("GET", "/api/v1/foo", base_url_override="https://other.api")
        call_kwargs = t._session.request.call_args
        assert call_kwargs.kwargs["url"] == "https://other.api/api/v1/foo"

    def test_passes_params(self):
        t = _make_transport()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        t._session.request = MagicMock(return_value=mock_resp)

        t.request("GET", "/models", params={"type": "chat"})
        assert t._session.request.call_args.kwargs["params"] == {"type": "chat"}


class TestSyncTransportErrors:
    def test_401_raises_authentication_error(self):
        t = _make_transport(max_retries=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"error": {"message": "bad key", "code": "auth"}}
        mock_resp.reason = "Unauthorized"
        t._session.request = MagicMock(return_value=mock_resp)

        with pytest.raises(AuthenticationError):
            t.request("GET", "/models")

    def test_429_raises_rate_limit_error(self):
        t = _make_transport(max_retries=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = {"error": {"message": "rate limited"}}
        mock_resp.reason = "Too Many Requests"
        t._session.request = MagicMock(return_value=mock_resp)

        with pytest.raises(RateLimitError):
            t.request("GET", "/models")

    def test_400_raises_api_error(self):
        t = _make_transport(max_retries=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"error": {"message": "bad request"}}
        mock_resp.reason = "Bad Request"
        t._session.request = MagicMock(return_value=mock_resp)

        with pytest.raises(APIError):
            t.request("POST", "/chat/completions", json={})

    def test_500_raises_api_error(self):
        t = _make_transport(max_retries=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": {"message": "internal error"}}
        mock_resp.reason = "Internal Server Error"
        t._session.request = MagicMock(return_value=mock_resp)

        with pytest.raises(APIError):
            t.request("GET", "/models")

    def test_error_string_format(self):
        t = _make_transport(max_retries=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"error": "plain string error"}
        mock_resp.reason = "Bad Request"
        t._session.request = MagicMock(return_value=mock_resp)

        with pytest.raises(APIError, match="plain string error"):
            t.request("GET", "/bad")

    def test_error_no_json(self):
        t = _make_transport(max_retries=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.json.side_effect = ValueError("not json")
        mock_resp.text = "Bad Gateway"
        mock_resp.reason = "Bad Gateway"
        t._session.request = MagicMock(return_value=mock_resp)

        with pytest.raises(APIError):
            t.request("GET", "/models")


class TestSyncTransportRetry:
    @patch("meganova.transport.time.sleep")
    def test_retries_on_network_error(self, mock_sleep):
        t = _make_transport(max_retries=2)

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"ok": True}

        t._session.request = MagicMock(
            side_effect=[
                requests.ConnectionError("refused"),
                ok_resp,
            ]
        )

        result = t.request("GET", "/models")
        assert result == {"ok": True}
        assert mock_sleep.call_count == 1

    @patch("meganova.transport.time.sleep")
    def test_exponential_backoff(self, mock_sleep):
        t = _make_transport(max_retries=2)

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"ok": True}

        t._session.request = MagicMock(
            side_effect=[
                requests.ConnectionError("fail1"),
                requests.ConnectionError("fail2"),
                ok_resp,
            ]
        )

        result = t.request("GET", "/models")
        assert result == {"ok": True}
        # 2^0=1, 2^1=2
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)

    @patch("meganova.transport.time.sleep")
    def test_exhausted_retries_raises(self, mock_sleep):
        t = _make_transport(max_retries=1)

        t._session.request = MagicMock(
            side_effect=[
                requests.ConnectionError("fail1"),
                requests.ConnectionError("fail2"),
            ]
        )

        with pytest.raises(MeganovaError, match="Network error"):
            t.request("GET", "/models")

    def test_no_retry_on_zero_retries(self):
        t = _make_transport(max_retries=0)
        t._session.request = MagicMock(
            side_effect=requests.ConnectionError("refused")
        )
        with pytest.raises(MeganovaError, match="Network error"):
            t.request("GET", "/models")
        assert t._session.request.call_count == 1

    def test_files_nulls_json_param(self):
        t = _make_transport()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"text": "hello"}
        t._session.request = MagicMock(return_value=mock_resp)

        t.request("POST", "/audio/transcriptions", files={"file": ("a.mp3", b"data", "audio/mpeg")}, data={"model": "whisper"})
        call_kwargs = t._session.request.call_args.kwargs
        assert call_kwargs["json"] is None
