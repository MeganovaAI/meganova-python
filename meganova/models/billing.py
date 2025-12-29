from pydantic import BaseModel
from typing import List, Optional

class InstanceBilling(BaseModel):
    vm_access_info_id: int
    running_hours: float
    fee_accumulated: float
    tax_rate: float
    fee_accumulated_taxed: float
    product_status: str

class BillingBalance(BaseModel):
    data: List[InstanceBilling]
    total_cost: float
    message: str
    status: str



