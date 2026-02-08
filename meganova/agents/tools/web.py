"""Web tools for agents.

Provides HTTP fetch capabilities.
"""

from typing import Optional

import requests

from .base import ToolDefinition


def _web_fetch(url: str, method: str = "GET", max_length: int = 5000) -> str:
    """Fetch a URL and return the response body (truncated)."""
    try:
        resp = requests.request(method, url, timeout=15)
        text = resp.text[:max_length]
        if len(resp.text) > max_length:
            text += f"\n... (truncated, {len(resp.text)} total chars)"
        return f"Status: {resp.status_code}\n\n{text}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


web_fetch_tool = ToolDefinition(
    name="web_fetch",
    description="Fetch a URL and return the response body. Useful for calling APIs or reading web pages.",
    func=_web_fetch,
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch",
            },
            "method": {
                "type": "string",
                "description": "HTTP method (GET, POST, etc.). Default: GET",
            },
        },
        "required": ["url"],
    },
)
