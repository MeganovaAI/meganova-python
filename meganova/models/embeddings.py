from pydantic import BaseModel
from typing import List, Optional


class Embedding(BaseModel):
    index: int
    embedding: list[float]
    object: str = "embedding"

    model_config = {"extra": "ignore"}


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int

    model_config = {"extra": "ignore"}


class EmbeddingResponse(BaseModel):
    data: List[Embedding]
    model: str
    object: str = "list"
    usage: Optional[EmbeddingUsage] = None

    model_config = {"extra": "ignore"}
