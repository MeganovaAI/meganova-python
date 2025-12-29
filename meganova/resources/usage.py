from typing import Optional
from ..transport import SyncTransport
from ..models.usage import UsageSummary

class UsageResource:
    def __init__(self, transport: SyncTransport):
        self._transport = transport

    def summary(
        self,
        *,
        from_: str,
        to: str,
        group_by: Optional[str] = None,
        team_id: Optional[int] = None, 
    ) -> dict:
        # Mapping to /usage/time_range
        params = {
            "start_time": from_,
            "end_time": to,
        }
        if team_id:
            params["team_id"] = team_id
            
        # Note: The actual API requires team_id. 
        # If it's not optional in the backend, the user must provide it.
        # For now, we pass kwargs to params.
        
        return self._transport.request("GET", "/usage/time_range", params=params)



