from typing import List, Optional
from ..transport import SyncTransport
from ..models.models_ import ModelInfo


class ModelsResource:
    def __init__(self, transport: SyncTransport):
        self._transport = transport

    def list(
        self,
        *,
        capability: Optional[str] = None,
        type: Optional[str] = None,
    ) -> List[ModelInfo]:
        data = self._transport.request("GET", "/models")
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

    def get(self, model_id: str) -> ModelInfo:
        data = self._transport.request("GET", f"/models/{model_id}")
        return ModelInfo(**data)
