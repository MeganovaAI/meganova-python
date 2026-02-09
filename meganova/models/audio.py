from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    text: str

    model_config = {"extra": "ignore"}
