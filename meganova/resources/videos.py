import time
from typing import Optional
from ..transport import SyncTransport
from ..models.videos import VideoGeneration
from ..errors import MeganovaError


class VideosResource:
    def __init__(self, transport: SyncTransport):
        self._transport = transport

    def generate(
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

        data = self._transport.request("POST", "/videos/generations", json=payload)
        result = VideoGeneration(**data)

        if wait:
            result = self._poll(result.id, poll_interval, timeout)

        return result

    def get(self, generation_id: str) -> VideoGeneration:
        data = self._transport.request("GET", f"/videos/generations/{generation_id}")
        return VideoGeneration(**data)

    def _poll(self, generation_id: str, interval: int, timeout: int) -> VideoGeneration:
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = self.get(generation_id)
            if result.status in ("completed", "failed"):
                return result
            time.sleep(interval)
        raise MeganovaError(f"Video generation {generation_id} timed out after {timeout}s")
