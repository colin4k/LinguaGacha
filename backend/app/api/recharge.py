from fastapi import APIRouter, Depends, HTTPException
from app.schemas.recharge import RechargeRequest, RechargeResponse, RechargeStatusResponse, RechargeNotifyRequest
from app.core.auth import get_current_user
from app.core.supabase_client import supabase
import uuid
import time

router = APIRouter()

@router.post("/recharge", response_model=RechargeResponse)
async def create_recharge(request: RechargeRequest, user_id: str = Depends(get_current_user)):
    try:
        order_id = str(uuid.uuid4())
        
        order_data = {
            "id": order_id,
            "user_id": user_id,
            "amount": request.amount,
            "status": "pending",
            "created_at": "now()"
        }
        
        response = supabase.table("recharge_orders").insert(order_data).execute()
        
        qr_code_url = f"https://example-payment-gateway.com/pay?order_id={order_id}&amount={request.amount}"
        
        return RechargeResponse(
            order_id=order_id,
            qr_code_url=qr_code_url,
            amount=request.amount
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recharge/status", response_model=RechargeStatusResponse)
async def get_recharge_status(order_id: str, user_id: str = Depends(get_current_user)):
    try:
        response = supabase.table("recharge_orders").select("*").eq("id", order_id).eq("user_id", user_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Order not found")
        
        order = response.data[0]
        return RechargeStatusResponse(
            order_id=order["id"],
            status=order["status"],
            amount=order["amount"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recharge/notify")
async def recharge_notify(request: RechargeNotifyRequest):
    try:
        if request.status == "completed":
            order_response = supabase.table("recharge_orders").select("*").eq("id", request.order_id).execute()
            if not order_response.data:
                raise HTTPException(status_code=404, detail="Order not found")
            
            order = order_response.data[0]
            
            supabase.table("recharge_orders").update({"status": "completed"}).eq("id", request.order_id).execute()
            
            profile_response = supabase.table("profiles").select("credits").eq("id", order["user_id"]).execute()
            current_credits = profile_response.data[0]["credits"] if profile_response.data else 0
            
            credits_to_add = int(request.amount * 10)
            supabase.table("profiles").upsert({
                "id": order["user_id"],
                "credits": current_credits + credits_to_add
            }).execute()
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))