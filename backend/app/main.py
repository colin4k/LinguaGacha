from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio

from app.api import recharge, translation, admin
from app.core.queue import task_queue
from app.services.translation_worker import start_translation_workers

@asynccontextmanager
async def lifespan(app: FastAPI):
    await task_queue.recover_pending_tasks()
    asyncio.create_task(start_translation_workers())
    yield

app = FastAPI(title="EPUB Translator API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recharge.router, prefix="/api")
app.include_router(translation.router, prefix="/api")
app.include_router(admin.router, prefix="/api/admin")

@app.get("/")
async def root():
    return {"message": "EPUB Translator API"}