import json
import time
from typing import Any, Iterator

import requests

from ..errors import MeganovaError, APIError, RateLimitError, AuthenticationError
from ..version import __version__


class CloudTransport:
    """HTTP transport for MegaNova Cloud Agent API.

    Unlike the main SDK transport, this does NOT use Bearer auth —
    the agent API key is embedded in the URL path.
    """

    def __init__(self, base_url: str, timeout: float, max_retries: int):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        params: dict | None = None,
        stream: bool = False,
    ) -> Any:
        url = f"{self.base_url}{path}"
        headers = {
            "User-Agent": f"meganova-python/{__version__}",
            "X-MN-SDK": "python",
        }
        if json_body is not None:
            headers["Content-Type"] = "application/json"

        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    json=json_body,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                    stream=stream,
                )

                if 200 <= response.status_code < 300:
                    if stream:
                        return response
                    return response.json()

                self._handle_error(response)

            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise MeganovaError(f"Network error: {exc}") from exc
                time.sleep(2 ** attempt)
                continue

        raise MeganovaError("Unexpected transport failure")

    def stream_sse(self, path: str, *, json_body: Any | None = None) -> Iterator[dict]:
        """Stream SSE events from a POST endpoint, yielding parsed data dicts."""
        response = self.request("POST", path, json_body=json_body, stream=True)
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
        finally:
            response.close()

    def _handle_error(self, response: requests.Response) -> None:
        status = response.status_code
        try:
            payload = response.json()
        except Exception:
            payload = {}

        error_data = payload.get("error", payload.get("detail", {}))
        if isinstance(error_data, str):
            message = error_data
            code = "unknown_error"
        elif isinstance(error_data, dict):
            message = error_data.get("message", "")
            if not message:
                message = str(payload) if payload else (response.reason or f"Error {status}")
            code = error_data.get("code") or "unknown_error"
        else:
            message = response.text or response.reason or f"Error {status}"
            code = "unknown_error"

        if status == 401:
            raise AuthenticationError(message, code=code, status=status)
        if status == 429:
            raise RateLimitError(message, code=code, status=status)
        if 400 <= status < 500:
            raise APIError(message, code=code, status=status)
        if 500 <= status < 600:
            raise APIError(message, code=code, status=status)

        raise MeganovaError(message, code=code, status=status)
