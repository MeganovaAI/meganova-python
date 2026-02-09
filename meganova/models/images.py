from pydantic import BaseModel
from typing import List, Optional


class GeneratedImage(BaseModel):
    b64_json: str
    revised_prompt: Optional[str] = None

    model_config = {"extra": "ignore"}


class ImageGenerationResponse(BaseModel):
    data: List[GeneratedImage]
