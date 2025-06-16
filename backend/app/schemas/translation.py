from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class TranslationTaskCreate(BaseModel):
    source_language: str
    target_language: str

class TranslationTaskResponse(BaseModel):
    id: str
    user_id: str
    source_language: str
    target_language: str
    status: str
    storage_path: Optional[str]
    translated_storage_path: Optional[str]
    created_at: datetime
    updated_at: datetime