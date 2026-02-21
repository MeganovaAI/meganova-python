"""Tests for AsyncTransport."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meganova.errors import APIError, AuthenticationError, MeganovaError, RateLimitError
from meganova.async_transport import AsyncTransport


def _make_async_transport(**kwargs):
    defaults = {
        "base_url": "https://api.meganova.ai/v1",
        "api_key": "test-key",
        "timeout": 60.0,
        "max_retries": 2,
        "user_agent": "meganova-python/0.3.0",
    }
    defaults.update(kwargs)
    return AsyncTransport(**defaults)


class TestAsyncTransportInit:
    def test_base_url_trailing_slash_stripped(self):
        t = _make_async_transport(base_url="https://api.meganova.ai/v1/")
        assert t.base_url == "https://api.meganova.ai/v1"

    def test_stores_api_key(self):
        t = _make_async_transport()
        assert t.api_key == "test-key"

    def test_sdk_header_is_python_async(self):
        # Verified in request headers, checked via test below
        t = _make_async_transport()
        assert t.user_agent == "meganova-python/0.3.0"


class TestAsyncTransportRequest:
    async def test_successful_json_response(self):
        t = _make_async_transport()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": "ok"}
        t._client.request = AsyncMock(return_value=mock_resp)

        result = await t.request("GET", "/models")
        assert result == {"result": "ok"}

    async def test_stream_returns_response_object(self):
        t = _make_async_transport()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        t._client.request = AsyncMock(return_value=mock_resp)

        result = await t.request("POST", "/chat/completions", json={"a": 1}, stream=True)
        assert result is mock_resp

    async def test_sends_auth_header(self):
        t = _make_async_transport()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        t._client.request = AsyncMock(return_value=mock_resp)

        await t.request("GET", "/models")
        call_kwargs = t._client.request.call_args
        headers = call_kwargs.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["X-MN-SDK"] == "python-async"


class TestAsyncTransportErrors:
    async def test_401_raises_authentication_error(self):
        t = _make_async_transport(max_retries=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"error": {"message": "bad key"}}
        t._client.request = AsyncMock(return_value=mock_resp)

        with pytest.raises(AuthenticationError):
            await t.request("GET", "/models")

    async def test_429_raises_rate_limit_error(self):
        t = _make_async_transport(max_retries=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = {"error": {"message": "rate limited"}}
        t._client.request = AsyncMock(return_value=mock_resp)

        with pytest.raises(RateLimitError):
            await t.request("GET", "/models")

    async def test_500_raises_api_error(self):
        t = _make_async_transport(max_retries=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": {"message": "internal"}}
        t._client.request = AsyncMock(return_value=mock_resp)

        with pytest.raises(APIError):
            await t.request("GET", "/models")

    async def test_error_string_format(self):
        t = _make_async_transport(max_retries=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"error": "string error"}
        t._client.request = AsyncMock(return_value=mock_resp)

        with pytest.raises(APIError, match="string error"):
            await t.request("GET", "/bad")

    async def test_error_no_json(self):
        t = _make_async_transport(max_retries=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.json.side_effect = ValueError("not json")
        t._client.request = AsyncMock(return_value=mock_resp)

        with pytest.raises(APIError):
            await t.request("GET", "/models")


class TestAsyncTransportRetry:
    async def test_retries_on_network_error(self):
        t = _make_async_transport(max_retries=1)

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"ok": True}

        t._client.request = AsyncMock(
            side_effect=[Exception("conn refused"), ok_resp]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await t.request("GET", "/models")
        assert result == {"ok": True}

    async def test_exhausted_retries_raises(self):
        t = _make_async_transport(max_retries=1)
        t._client.request = AsyncMock(
            side_effect=[Exception("fail1"), Exception("fail2")]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(MeganovaError, match="Network error"):
                await t.request("GET", "/models")

    async def test_does_not_retry_meganova_errors(self):
        t = _make_async_transport(max_retries=2)
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"error": {"message": "bad key"}}
        t._client.request = AsyncMock(return_value=mock_resp)

        with pytest.raises(AuthenticationError):
            await t.request("GET", "/models")
        # Should only be called once — no retry
        assert t._client.request.call_count == 1


class TestAsyncTransportSSE:
    async def test_stream_sse_yields_chunks(self):
        t = _make_async_transport()
        chunk1 = {"id": "c1", "choices": [{"delta": {"content": "Hi"}}]}
        chunk2 = {"id": "c2", "choices": [{"delta": {"content": " there"}}]}

        lines = [
            f"data: {json.dumps(chunk1)}",
            f"data: {json.dumps(chunk2)}",
            "data: [DONE]",
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = lambda: _async_iter(lines)

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)
        t._client.stream = MagicMock(return_value=mock_stream_cm)

        chunks = []
        async for c in t.stream_sse("POST", "/chat/completions", json={"model": "x"}):
            chunks.append(c)

        assert len(chunks) == 2
        assert chunks[0]["id"] == "c1"

    async def test_stream_sse_skips_empty_lines(self):
        t = _make_async_transport()
        chunk = {"id": "c1"}

        lines = [
            "",
            "   ",
            f"data: {json.dumps(chunk)}",
            "data: [DONE]",
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = lambda: _async_iter(lines)

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)
        t._client.stream = MagicMock(return_value=mock_stream_cm)

        chunks = []
        async for c in t.stream_sse("POST", "/chat/completions"):
            chunks.append(c)

        assert len(chunks) == 1


class TestAsyncTransportContextManager:
    async def test_close(self):
        t = _make_async_transport()
        t._client.aclose = AsyncMock()
        await t.close()
        t._client.aclose.assert_called_once()

    async def test_aenter_returns_self(self):
        t = _make_async_transport()
        result = await t.__aenter__()
        assert result is t
        t._client.aclose = AsyncMock()
        await t.__aexit__(None, None, None)


async def _async_iter(items):
    for item in items:
        yield item
