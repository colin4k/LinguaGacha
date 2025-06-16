from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from typing import List
import os
import uuid
import aiofiles
from app.schemas.translation import TranslationTaskCreate, TranslationTaskResponse
from app.core.auth import get_current_user
from app.core.supabase_client import supabase
from app.core.queue import task_queue
from app.core.config import settings

router = APIRouter()

@router.post("/translations/upload")
async def upload_translation_file(
    file: UploadFile = File(...),
    source_language: str = "auto",
    target_language: str = "zh",
    user_id: str = Depends(get_current_user)
):
    try:
        profile_response = supabase.table("profiles").select("credits").eq("id", user_id).execute()
        if not profile_response.data or profile_response.data[0]["credits"] < 10:
            raise HTTPException(status_code=400, detail="Insufficient credits")
        
        user_upload_dir = os.path.join(settings.UPLOAD_DIR, user_id)
        os.makedirs(user_upload_dir, exist_ok=True)
        
        file_extension = os.path.splitext(file.filename)[1]
        file_id = str(uuid.uuid4())
        file_path = os.path.join(user_upload_dir, f"{file_id}{file_extension}")
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        task_data = {
            "id": file_id,
            "user_id": user_id,
            "source_language": source_language,
            "target_language": target_language,
            "status": "pending",
            "original_filename": file.filename,
            "storage_path": file_path,
            "created_at": "now()"
        }
        
        response = supabase.table("translation_tasks").insert(task_data).execute()
        
        await task_queue.put(file_id)
        
        return {"task_id": file_id, "message": "File uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/translations", response_model=List[TranslationTaskResponse])
async def get_translations(user_id: str = Depends(get_current_user)):
    try:
        response = supabase.table("translation_tasks").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/translations/{task_id}/download")
async def download_translation(task_id: str, user_id: str = Depends(get_current_user)):
    try:
        response = supabase.table("translation_tasks").select("*").eq("id", task_id).eq("user_id", user_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task = response.data[0]
        
        if task["status"] != "completed" or not task["translated_storage_path"]:
            raise HTTPException(status_code=400, detail="Translation not completed")
        
        if not os.path.exists(task["translated_storage_path"]):
            raise HTTPException(status_code=404, detail="File not found")
        
        filename = f"translated_{task['original_filename']}"
        return FileResponse(
            path=task["translated_storage_path"],
            filename=filename,
            media_type='application/octet-stream'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))