from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from app.core.supabase_client import supabase
import requests

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        response = supabase.auth.get_user(token)
        if response.user:
            return response.user.id
        else:
            raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_admin_user(user_id: str = Depends(get_current_user)):
    try:
        response = supabase.table("profiles").select("is_admin").eq("id", user_id).execute()
        if response.data and response.data[0].get("is_admin"):
            return user_id
        else:
            raise HTTPException(status_code=403, detail="Admin access required")
    except Exception as e:
        raise HTTPException(status_code=403, detail="Admin access required")