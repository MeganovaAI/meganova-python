from typing import Optional

from .config import PRODUCTION_API_URL, DEFAULT_TIMEOUT, MAX_RETRIES
from .transport import SyncTransport
from .version import __version__
from .resources.chat import Chat
from .resources.models_ import ModelsResource
from .resources.usage import UsageResource
from .resources.billing import BillingResource

class MegaNova:
    def __init__(
        self,
        api_key: str,
        base_url: str = PRODUCTION_API_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        region: str = "auto",
        user_agent_extra: Optional[str] = None,
    ):
        if not api_key:
            raise ValueError("api_key is required")

        user_agent = f"meganova-python/{__version__}"
        if user_agent_extra:
            user_agent += f" {user_agent_extra}"

        self._transport = SyncTransport(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            user_agent=user_agent,
        )

        self.chat = Chat(self._transport)
        self.models = ModelsResource(self._transport)
        self.usage = UsageResource(self._transport)
        self.billing = BillingResource(self._transport)
