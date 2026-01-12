"""
FastAPI зависимости для RuQualBench Server.
"""

import os

from server.services.evaluator import EvaluatorService
from server.services.job_store import JobStore


# Синглтоны для сервисов
_evaluator_service: EvaluatorService = None
_job_store: JobStore = None


def get_evaluator() -> EvaluatorService:
    """Возвращает синглтон EvaluatorService."""
    global _evaluator_service
    if _evaluator_service is None:
        _evaluator_service = EvaluatorService()
    return _evaluator_service


def get_job_store() -> JobStore:
    """Возвращает синглтон JobStore."""
    global _job_store
    if _job_store is None:
        # Получаем настройку из переменной окружения, установленной в __main__.py
        retention_hours = int(os.getenv("_SERVER_JOB_RETENTION_HOURS", "24"))
        _job_store = JobStore(retention_hours=retention_hours)
    return _job_store


def reset_services():
    """Сбрасывает синглтоны (для тестов)."""
    global _evaluator_service, _job_store
    _evaluator_service = None
    _job_store = None
