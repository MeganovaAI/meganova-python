from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class ModelInfo(BaseModel):
    id: str
    model_name: str
    description: Optional[str] = None
    input_price: float
    output_price: float
    hosting_in: Optional[str] = None
    parameter_size: Optional[str] = None
    huggingface_url: Optional[str] = None
    nebulablock_url: Optional[str] = None
    readme: Optional[str] = None

class ModelListResponse(BaseModel):
    data: Dict[str, List[ModelInfo]]
    message: str
    status: str
