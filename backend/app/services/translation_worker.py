import asyncio
import os
from app.core.queue import task_queue
from app.core.supabase_client import supabase
from app.core.config import settings

async def process_translation_task(task_id: str):
    try:
        supabase.table("translation_tasks").update({
            "status": "processing",
            "updated_at": "now()"
        }).eq("id", task_id).execute()
        
        task_response = supabase.table("translation_tasks").select("*").eq("id", task_id).execute()
        if not task_response.data:
            return
        
        task = task_response.data[0]
        input_file_path = task["storage_path"]
        
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file not found: {input_file_path}")
        
        config_response = supabase.table("llm_config").select("*").limit(1).execute()
        if not config_response.data:
            raise Exception("LLM configuration not found")
        
        llm_config = config_response.data[0]
        
        user_dir = os.path.dirname(input_file_path)
        output_filename = f"translated_{task['original_filename']}"
        output_file_path = os.path.join(user_dir, output_filename)
        
        try:
            from api import translate_epub_file
            
            success = translate_epub_file(
                input_file_path,
                output_file_path,
                task["source_language"],
                task["target_language"],
                llm_config["base_url"],
                llm_config["api_key"],
                llm_config["model"]
            )
            
            if success:
                supabase.table("translation_tasks").update({
                    "status": "completed",
                    "translated_storage_path": output_file_path,
                    "updated_at": "now()"
                }).eq("id", task_id).execute()
                
                profile_response = supabase.table("profiles").select("credits").eq("id", task["user_id"]).execute()
                current_credits = profile_response.data[0]["credits"] if profile_response.data else 0
                
                supabase.table("profiles").update({
                    "credits": max(0, current_credits - 10)
                }).eq("id", task["user_id"]).execute()
            else:
                raise Exception("Translation failed")
                
        except Exception as translation_error:
            supabase.table("translation_tasks").update({
                "status": "failed",
                "error_message": str(translation_error),
                "updated_at": "now()"
            }).eq("id", task_id).execute()
            
    except Exception as e:
        print(f"Error processing task {task_id}: {e}")
        try:
            supabase.table("translation_tasks").update({
                "status": "failed",
                "error_message": str(e),
                "updated_at": "now()"
            }).eq("id", task_id).execute()
        except:
            pass
    finally:
        task_queue.task_done(task_id)

async def translation_worker():
    while True:
        try:
            task_id = await task_queue.get()
            await process_translation_task(task_id)
        except Exception as e:
            print(f"Worker error: {e}")
            await asyncio.sleep(1)

async def start_translation_workers():
    workers = []
    for i in range(settings.TRANSLATION_WORKERS):
        worker = asyncio.create_task(translation_worker())
        workers.append(worker)
    
    await asyncio.gather(*workers)