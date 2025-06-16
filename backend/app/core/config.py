import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    SUPABASE_URL: str = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY")
    TRANSLATION_WORKERS: int = int(os.getenv("TRANSLATION_WORKERS", 3))
    UPLOAD_DIR: str = "uploads"
    
settings = Settings()