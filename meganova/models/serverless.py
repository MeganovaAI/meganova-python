from pydantic import BaseModel, Field
from typing import List, Optional


class ServerlessModel(BaseModel):
    model_name: str
    model_alias: Optional[str] = None
    cost_per_1k_input: Optional[float] = None
    cost_per_1k_output: Optional[float] = None
    modality: Optional[str] = None
    tier2_plus_require: Optional[bool] = Field(None, alias="tier2+_require")
    restrictions: Optional[str] = None

    model_config = {"extra": "ignore", "populate_by_name": True}


class ServerlessModelsResponse(BaseModel):
    models: List[ServerlessModel]
    count: int
