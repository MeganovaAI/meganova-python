from typing import Iterator, Optional, List, Union, Dict
import json
from ..transport import SyncTransport
from ..models.chat import ChatResponse, ChatStreamChunk

class ChatResource:
    def __init__(self, transport: SyncTransport):
        self._transport = transport

    def create(
        self,
        *,
        messages: List[Dict[str, str]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatResponse, Iterator[ChatStreamChunk]]:
        payload = {
            "messages": messages,
            "model": model,
            "stream": stream,
            **kwargs,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if stream:
            return self._stream_request(payload)
        
        data = self._transport.request("POST", "/chat/completions", json=payload)
        return ChatResponse(**data)

    def _stream_request(self, payload: dict) -> Iterator[ChatStreamChunk]:
        response = self._transport.request("POST", "/chat/completions", json=payload, stream=True)
        
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

    def stream(
        self,
        *,
        messages: List[Dict[str, str]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Iterator[ChatStreamChunk]:
        """Explicit streaming method that only returns an iterator."""
        return self.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )



