import asyncio
from typing import Set
from app.core.supabase_client import supabase

class TaskQueue:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.processing_tasks: Set[str] = set()
    
    async def put(self, task_id: str):
        await self.queue.put(task_id)
    
    async def get(self) -> str:
        task_id = await self.queue.get()
        self.processing_tasks.add(task_id)
        return task_id
    
    def task_done(self, task_id: str):
        self.processing_tasks.discard(task_id)
        self.queue.task_done()
    
    async def recover_pending_tasks(self):
        try:
            response = supabase.table("translation_tasks").select("id").in_("status", ["pending", "processing"]).execute()
            for task in response.data:
                await self.put(task["id"])
        except Exception as e:
            print(f"Error recovering tasks: {e}")

task_queue = TaskQueue()