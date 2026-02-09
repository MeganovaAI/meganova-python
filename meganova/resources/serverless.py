from typing import List
from ..transport import SyncTransport
from ..models.serverless import ServerlessModel, ServerlessModelsResponse


class ServerlessResource:
    def __init__(self, transport: SyncTransport):
        self._transport = transport

    def list_models(self, modality: str = "text_generation") -> ServerlessModelsResponse:
        """List available serverless models filtered by modality.

        Args:
            modality: Filter by modality, e.g. "text_generation" or "text_to_image".

        Returns:
            ServerlessModelsResponse with models list and count.
        """
        # The serverless endpoint uses /api/v1/ instead of the standard /v1/ base.
        data = self._transport.request(
            "GET",
            "/api/v1/serverless/models/filter",
            params={"modality": modality},
            base_url_override=self._transport.base_url.replace("/v1", ""),
        )
        # Response: {"data": {"models": [...], "count": N}}
        inner = data.get("data", data)
        return ServerlessModelsResponse(**inner)
