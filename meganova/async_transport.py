from typing import Any, AsyncIterator
import json

from .errors import MeganovaError, APIError, RateLimitError, AuthenticationError
from .version import __version__


class AsyncTransport:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float,
        max_retries: int,
        user_agent: str,
    ):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self._client = httpx.AsyncClient(timeout=timeout)

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: dict | None = None,
        stream: bool = False,
    ) -> Any:
        import asyncio

        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": self.user_agent,
            "X-MN-SDK": "python-async",
        }

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.request(
                    method=method,
                    url=url,
                    json=json,
                    params=params,
                    headers=headers,
                )

                if 200 <= response.status_code < 300:
                    if stream:
                        return response
                    return response.json()

                self._handle_error(response)

            except (MeganovaError, APIError, AuthenticationError, RateLimitError):
                raise
            except Exception as exc:
                if attempt >= self.max_retries:
                    raise MeganovaError(f"Network error: {exc}") from exc
                await asyncio.sleep(2 ** attempt)
                continue

        raise MeganovaError("Unexpected transport failure")

    async def stream_sse(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
    ) -> AsyncIterator:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": self.user_agent,
            "X-MN-SDK": "python-async",
        }

        async with self._client.stream(
            method, url, json=json, headers=headers,
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                self._handle_error(response)

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    import json as json_mod
                    yield json_mod.loads(data_str)
                except Exception:
                    continue

    def _handle_error(self, response) -> None:
        status = response.status_code
        try:
            payload = response.json()
        except Exception:
            payload = {}

        error_data = payload.get("error", {})
        if isinstance(error_data, str):
            message = error_data
            code = "unknown_error"
        elif isinstance(error_data, dict):
            message = error_data.get("message")
            if not message:
                message = str(payload) if payload else f"Error {status}"
            code = error_data.get("code") or "unknown_error"
        else:
            message = f"Error {status}"
            code = "unknown_error"

        if status == 401:
            raise AuthenticationError(message, code=code, status=status)
        if status == 429:
            raise RateLimitError(message, code=code, status=status)
        if 400 <= status < 600:
            raise APIError(message, code=code, status=status)
        raise MeganovaError(message, code=code, status=status)

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
