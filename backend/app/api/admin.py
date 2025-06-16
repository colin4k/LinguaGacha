from fastapi import APIRouter, Depends, HTTPException
from app.schemas.admin import LLMConfig, LLMConfigResponse
from app.core.auth import get_admin_user
from app.core.supabase_client import supabase

router = APIRouter()

@router.get("/llm-settings", response_model=LLMConfigResponse)
async def get_llm_settings(admin_user_id: str = Depends(get_admin_user)):
    try:
        response = supabase.table("llm_config").select("base_url, model").limit(1).execute()
        
        if not response.data:
            return LLMConfigResponse(base_url="", model="")
        
        config = response.data[0]
        return LLMConfigResponse(
            base_url=config["base_url"],
            model=config["model"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/llm-settings")
async def update_llm_settings(config: LLMConfig, admin_user_id: str = Depends(get_admin_user)):
    try:
        existing_response = supabase.table("llm_config").select("id").limit(1).execute()
        
        config_data = {
            "base_url": config.base_url,
            "api_key": config.api_key,
            "model": config.model,
            "updated_at": "now()"
        }
        
        if existing_response.data:
            config_data["id"] = existing_response.data[0]["id"]
            response = supabase.table("llm_config").upsert(config_data).execute()
        else:
            response = supabase.table("llm_config").insert(config_data).execute()
        
        return {"message": "LLM settings updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))