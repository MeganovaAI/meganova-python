import json
from typing import AsyncIterator, Optional, Union, List, Dict, Any

from ..async_transport import AsyncTransport
from ..models.chat import ChatResponse, ChatStreamChunk
from ..models.embeddings import EmbeddingResponse
from ..models.images import ImageGenerationResponse
from ..models.videos import VideoGeneration
from ..models.models_ import ModelInfo
from ..models.audio import TranscriptionResponse
from ..errors import MeganovaError


class AsyncCompletions:
    def __init__(self, transport: AsyncTransport):
        self._transport = transport

    async def create(
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
    ) -> Union[ChatResponse, AsyncIterator[ChatStreamChunk]]:
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

        data = await self._transport.request("POST", "/chat/completions", json=payload)
        return ChatResponse(**data)

    async def _stream_request(self, payload: dict) -> AsyncIterator[ChatStreamChunk]:
        async for chunk in self._transport.stream_sse("POST", "/chat/completions", json=payload):
            yield ChatStreamChunk(**chunk)


class AsyncChat:
    def __init__(self, transport: AsyncTransport):
        self.completions = AsyncCompletions(transport)


class AsyncEmbeddingsResource:
    def __init__(self, transport: AsyncTransport):
        self._transport = transport

    async def create(
        self,
        *,
        input: Union[str, List[str]],
        model: str,
        encoding_format: Optional[str] = None,
    ) -> EmbeddingResponse:
        payload = {
            "input": input,
            "model": model,
        }
        if encoding_format is not None:
            payload["encoding_format"] = encoding_format

        data = await self._transport.request("POST", "/embeddings", json=payload)
        return EmbeddingResponse(**data)


class AsyncImagesResource:
    def __init__(self, transport: AsyncTransport):
        self._transport = transport

    async def generate(
        self,
        prompt: str,
        *,
        model: str,
        n: int = 1,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        response_format: Optional[str] = None,
        style: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> ImageGenerationResponse:
        payload = {
            "model": model,
            "prompt": prompt,
            "n": n,
        }
        if size is not None:
            payload["size"] = size
        elif width is not None and height is not None:
            payload["size"] = f"{width}x{height}"

        for key, value in [
            ("quality", quality),
            ("response_format", response_format),
            ("style", style),
            ("num_steps", num_steps),
            ("guidance_scale", guidance_scale),
            ("seed", seed),
        ]:
            if value is not None:
                payload[key] = value

        payload.update(kwargs)

        data = await self._transport.request("POST", "/images/generations", json=payload)
        return ImageGenerationResponse(**data)


class AsyncVideosResource:
    def __init__(self, transport: AsyncTransport):
        self._transport = transport

    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        image_url: Optional[str] = None,
        size: str = "1280x720",
        n_seconds: int = 5,
        aspect_ratio: Optional[str] = None,
        seed: Optional[int] = None,
        wait: bool = False,
        poll_interval: int = 5,
        timeout: int = 600,
    ) -> VideoGeneration:
        import asyncio

        payload = {
            "prompt": prompt,
            "model": model,
            "size": size,
            "n_seconds": n_seconds,
        }
        if image_url is not None:
            payload["image_url"] = image_url
        if aspect_ratio is not None:
            payload["aspect_ratio"] = aspect_ratio
        if seed is not None:
            payload["seed"] = seed

        data = await self._transport.request("POST", "/videos/generations", json=payload)
        result = VideoGeneration(**data)

        if wait:
            result = await self._poll(result.id, poll_interval, timeout)

        return result

    async def get(self, generation_id: str) -> VideoGeneration:
        data = await self._transport.request("GET", f"/videos/generations/{generation_id}")
        return VideoGeneration(**data)

    async def _poll(self, generation_id: str, interval: int, timeout: int) -> VideoGeneration:
        import asyncio
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            result = await self.get(generation_id)
            if result.status in ("completed", "failed"):
                return result
            await asyncio.sleep(interval)
        raise MeganovaError(f"Video generation {generation_id} timed out after {timeout}s")


class AsyncModelsResource:
    def __init__(self, transport: AsyncTransport):
        self._transport = transport

    async def list(
        self,
        *,
        capability: Optional[str] = None,
        type: Optional[str] = None,
    ) -> List[ModelInfo]:
        data = await self._transport.request("GET", "/models")
        raw_models = data.get("data", [])
        models = [ModelInfo(**m) for m in raw_models]

        if capability:
            models = [
                m for m in models
                if m.capabilities and m.capabilities.get(capability)
            ]
        if type:
            models = [
                m for m in models
                if m.tags and type in m.tags
            ]

        return models

    async def get(self, model_id: str) -> ModelInfo:
        data = await self._transport.request("GET", f"/models/{model_id}")
        return ModelInfo(**data)


class AsyncAudioResource:
    def __init__(self, transport: AsyncTransport):
        self._transport = transport

    async def transcribe(
        self,
        file_path: str,
        *,
        model: str = "Systran/faster-whisper-large-v3",
    ) -> TranscriptionResponse:
        import httpx
        from pathlib import Path

        path = Path(file_path)
        url = f"{self._transport.base_url}/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {self._transport.api_key}",
            "User-Agent": self._transport.user_agent,
            "X-MN-SDK": "python-async",
        }

        with open(path, "rb") as f:
            files = {"file": (path.name, f, "audio/mpeg")}
            data = {"model": model}
            async with httpx.AsyncClient(timeout=self._transport.timeout) as client:
                response = await client.post(url, files=files, data=data, headers=headers)
                response.raise_for_status()
                return TranscriptionResponse(**response.json())
