"""
RuQualBench Server - FastAPI приложение.

Сервер для оценки качества русского языка в ответах LLM.
"""

import logging
import os
from contextlib import asynccontextmanager

import litellm
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server import __version__
from server.dependencies import get_job_store
from server.routers import evaluate_router, jobs_router

# Загружаем переменные окружения
load_dotenv()

# Настройка litellm
litellm.ssl_verify = False

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Отключаем verbose логи от litellm
for logger_name in ['litellm', 'LiteLLM']:
    litellm_logger = logging.getLogger(logger_name)
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    # Startup
    logger.info(f"Запуск RuQualBench Server v{__version__}")
    
    # Запускаем фоновую задачу очистки старых jobs
    job_store = get_job_store()
    await job_store.start_cleanup_task()
    
    yield
    
    # Shutdown
    logger.info("Остановка RuQualBench Server")
    await job_store.stop_cleanup_task()


# Создаём приложение
app = FastAPI(
    title="RuQualBench Server",
    description="""
API для оценки качества русского языка в ответах LLM.

## Версии оценки

### V1 - Подсчёт ошибок по категориям
Возвращает количество критических, обычных и дополнительных ошибок с описаниями.

### V2 - Детальный список ошибок
Возвращает список ошибок с позициями в тексте, уровнями и типами.

## Режимы обработки

- **Одиночная оценка** - синхронная оценка одного ответа
- **Синхронный батч** - оценка нескольких ответов, клиент ждёт результат
- **Асинхронный батч** - создание задачи с job_id для отслеживания прогресса
    """,
    version=__version__,
    lifespan=lifespan
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роутеры
app.include_router(evaluate_router, prefix="/api")
app.include_router(jobs_router, prefix="/api")


@app.get("/", tags=["health"])
async def root():
    """Корневой эндпоинт - информация о сервере."""
    return {
        "name": "RuQualBench Server",
        "version": __version__,
        "status": "running"
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Проверка здоровья сервера."""
    return {"status": "healthy"}


def create_app() -> FastAPI:
    """Фабрика приложения (для тестов и gunicorn)."""
    return app
