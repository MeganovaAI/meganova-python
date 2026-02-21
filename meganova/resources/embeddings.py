from typing import Optional, Union, List
from ..transport import SyncTransport
from ..models.embeddings import EmbeddingResponse


class EmbeddingsResource:
    def __init__(self, transport: SyncTransport):
        self._transport = transport

    def create(
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

        data = self._transport.request("POST", "/embeddings", json=payload)
        return EmbeddingResponse(**data)
