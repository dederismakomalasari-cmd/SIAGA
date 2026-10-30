"""Entry point SIAGA."""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from .config import settings
from .database import init_db
from .api.routes import router
from .pipeline import start_all_workers

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND = BASE_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("[SIAGA] Database ready.")
    task = asyncio.create_task(start_all_workers())
    print("[SIAGA] Workers started.")
    yield
    task.cancel()


app = FastAPI(title="SIAGA — Sistem Identifikasi dan Antisipasi Gangguan",
              version="1.0.0", lifespan=lifespan)

app.include_router(router)

# static
app.mount("/snapshots", StaticFiles(directory=str(settings.snapshot_dir)), name="snapshots")
app.mount("/ui", StaticFiles(directory=str(FRONTEND), html=True), name="ui")


@app.get("/")
def root():
    return RedirectResponse("/ui/login.html")