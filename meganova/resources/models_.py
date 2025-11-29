from typing import List
from ..transport import SyncTransport
from ..models.models_ import ModelInfo

class ModelsResource:
    def __init__(self, transport: SyncTransport):
        self._transport = transport

    def list(self) -> List[ModelInfo]:
        data = self._transport.request("GET", "/serverless/models")
        # The API returns {"data": {"models": [...]}, ...}
        raw_models = data.get("data", {}).get("models", [])
        return [ModelInfo(**m) for m in raw_models]

