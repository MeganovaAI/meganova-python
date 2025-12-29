from typing import Optional
from ..transport import SyncTransport
from ..models.billing import BillingBalance

class BillingResource:
    def __init__(self, transport: SyncTransport):
        self._transport = transport

    def get_user_instance_billings(self) -> BillingBalance:
        data = self._transport.request("GET", "/billing/user-instance-billings")
        return BillingBalance(**data)



