from pydantic import BaseModel
from typing import Optional

class RechargeRequest(BaseModel):
    amount: float

class RechargeResponse(BaseModel):
    order_id: str
    qr_code_url: str
    amount: float

class RechargeStatusResponse(BaseModel):
    order_id: str
    status: str
    amount: float

class RechargeNotifyRequest(BaseModel):
    order_id: str
    status: str
    amount: float
    signature: Optional[str]