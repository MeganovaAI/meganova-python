from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class UsageDetail(BaseModel):
    # Define fields based on actual API response
    # Placeholder for now as exact structure might vary
    date: str
    model: str
    tokens: int
    cost: float

class UsageSummary(BaseModel):
    data: List[UsageDetail]
    message: str
    status: str


