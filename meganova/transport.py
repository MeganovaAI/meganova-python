import requests
from typing import Any, Dict, Optional
import time

from .errors import MeganovaError, APIError, RateLimitError, AuthenticationError
from .version import __version__

class SyncTransport:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float,
        max_retries: int,
        user_agent: str
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self._session = requests.Session()

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: dict | None = None,
        stream: bool = False,
    ) -> Any:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": self.user_agent,
            "X-MN-SDK": "python",
        }

        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    json=json,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                    stream=stream,
                )
                
                if 200 <= response.status_code < 300:
                    if stream:
                        return response
                    return response.json()

                # Handle errors
                self._handle_error(response)

            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise MeganovaError(f"Network error: {exc}") from exc
                # Exponential backoff
                time.sleep(2 ** attempt)
                continue

        raise MeganovaError("Unexpected transport failure")

    def _handle_error(self, response: requests.Response) -> None:
        status = response.status_code
        try:
            payload = response.json()
        except Exception:
            payload = {}

        # Try to extract error message from common patterns
        error_data = payload.get("error", {})
        if isinstance(error_data, str):
             message = error_data
             code = "unknown_error"
        elif isinstance(error_data, dict):
            message = error_data.get("message") or str(payload)
            code = error_data.get("code") or "unknown_error"
        else:
            message = response.text or "Unknown error"
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


