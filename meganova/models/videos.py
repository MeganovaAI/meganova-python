from pydantic import BaseModel
from typing import Optional


class VideoGeneration(BaseModel):
    id: str
    object: str = "video.generation"
    created_at: int
    status: str
    model: str
    url: Optional[str] = None
    error: Optional[dict] = None
    estimated_cost: Optional[float] = None

    model_config = {"extra": "ignore"}
