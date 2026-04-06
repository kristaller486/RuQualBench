import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.backend.config import load_settings
from web.backend.database import Database
from web.backend.auth import AdminAuthMiddleware
from web.backend.services.log_parser import LogStore
from web.backend.services.sync_service import SyncService
from web.backend.routers import leaderboard, results, admin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings

    # Инициализация БД
    db = Database(settings.database_url)
    await db.connect()
    await db.ensure_default_tag_types()
    app.state.db = db

    # Парсинг логов
    log_store = LogStore(settings.logs_v2_dir)
    log_store.reload()
    app.state.log_store = log_store
    logger.info(f"Загружено моделей: {len(log_store.get_model_ids())}")

    # Регистрация моделей в БД
    for mid in log_store.get_model_ids():
        name = log_store.get_model_name(mid)
        await db.ensure_model(mid, name)

    # Сервис синхронизации
    app.state.sync_service = SyncService(settings.repo_path)

    yield

    await db.close()


app = FastAPI(title="RuQualBench UI", version="0.1.0", lifespan=lifespan)

# CORS для локальной разработки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth middleware
settings = load_settings()
app.add_middleware(AdminAuthMiddleware, settings=settings)

# Роуты
app.include_router(leaderboard.router)
app.include_router(results.router)
app.include_router(admin.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Статические файлы React (если собраны)
static_dir = Path(__file__).parent.parent / "frontend" / "dist"
if static_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="static-assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = static_dir / full_path
        if file_path.is_file() and ".." not in full_path:
            return FileResponse(file_path)
        return FileResponse(static_dir / "index.html")
