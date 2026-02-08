from typing import Iterator, Optional, List, Union, Dict, Any
import json
from ..transport import SyncTransport
from ..models.chat import ChatResponse, ChatStreamChunk


class Completions:
    def __init__(self, transport: SyncTransport):
        self._transport = transport

    def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        **kwargs,
    ) -> Union[ChatResponse, Iterator[ChatStreamChunk]]:
        payload: Dict[str, Any] = {
            "messages": messages,
            "model": model,
            "stream": stream,
            **kwargs,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        if stream:
            return self._stream_request(payload)

        data = self._transport.request("POST", "/chat/completions", json=payload)
        return ChatResponse(**data)

    def _stream_request(self, payload: dict) -> Iterator[ChatStreamChunk]:
        response = self._transport.request(
            "POST", "/chat/completions", json=payload, stream=True
        )

        for line in response.iter_lines():
            if not line:
                continue

            line_text = line.decode("utf-8").strip()
            if line_text.startswith("data: "):
                data_str = line_text[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data_json = json.loads(data_str)
                    yield ChatStreamChunk(**data_json)
                except json.JSONDecodeError:
                    continue


class Chat:
    def __init__(self, transport: SyncTransport):
        self.completions = Completions(transport)
