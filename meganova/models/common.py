# Common models shared across resources can go here
from pydantic import BaseModel

class BaseResponse(BaseModel):
    status: str
    message: str

