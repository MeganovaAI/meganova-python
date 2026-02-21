from pydantic import BaseModel
from typing import List, Optional


class GeneratedImage(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None

    model_config = {"extra": "ignore"}


class ImageGenerationResponse(BaseModel):
    data: List[GeneratedImage]
    created: Optional[int] = None
