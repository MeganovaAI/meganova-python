from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ModelInfo(BaseModel):
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    created: Optional[int] = None
    pricing: Optional[Dict[str, str]] = None
    capabilities: Optional[Dict[str, bool]] = None
    
    # Allow extra fields since the API returns many more
    model_config = {"extra": "ignore"}

class ModelListResponse(BaseModel):
    data: List[ModelInfo]
