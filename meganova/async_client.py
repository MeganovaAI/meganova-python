from typing import Optional

from .config import PRODUCTION_API_URL, DEFAULT_TIMEOUT, MAX_RETRIES
from .async_transport import AsyncTransport
from .version import __version__
from .resources.async_resources import (
    AsyncChat,
    AsyncEmbeddingsResource,
    AsyncImagesResource,
    AsyncVideosResource,
    AsyncModelsResource,
    AsyncAudioResource,
)


class AsyncMegaNova:
    """Async MegaNova inference API client.

    Usage:
        async with AsyncMegaNova(api_key="...") as client:
            response = await client.chat.completions.create(...)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = PRODUCTION_API_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        user_agent_extra: Optional[str] = None,
    ):
        if not api_key:
            raise ValueError("api_key is required")

        user_agent = f"meganova-python/{__version__}"
        if user_agent_extra:
            user_agent += f" {user_agent_extra}"

        self._transport = AsyncTransport(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            user_agent=user_agent,
        )

        self.chat = AsyncChat(self._transport)
        self.embeddings = AsyncEmbeddingsResource(self._transport)
        self.images = AsyncImagesResource(self._transport)
        self.videos = AsyncVideosResource(self._transport)
        self.models = AsyncModelsResource(self._transport)
        self.audio = AsyncAudioResource(self._transport)

    async def close(self):
        await self._transport.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
