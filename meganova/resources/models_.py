from typing import List
from ..transport import SyncTransport
from ..models.models_ import ModelInfo

class ModelsResource:
    def __init__(self, transport: SyncTransport):
        self._transport = transport

    def list(self) -> List[ModelInfo]:
        data = self._transport.request("GET", "/models")
        # The API returns {"data": [...]}
        raw_models = data.get("data", [])
        return [ModelInfo(**m) for m in raw_models]
