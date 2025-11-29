from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union, Dict

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[TokenUsage] = None
    system_fingerprint: Optional[str] = None

class ChatStreamChunkChoice(BaseModel):
    index: int
    delta: Dict[str, str]
    finish_reason: Optional[str] = None

class ChatStreamChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatStreamChunkChoice]

