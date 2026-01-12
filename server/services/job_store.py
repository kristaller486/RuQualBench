"""
Хранилище асинхронных задач.

In-memory реализация для MVP. Может быть расширена для использования Redis или БД.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from server.models import BatchSummary, JobInfoResponse, JobProgress, JobStatus

logger = logging.getLogger(__name__)


class Job:
    """Внутреннее представление задачи."""
    
    def __init__(self, version: Literal["v1", "v2"], total_items: int):
        self.job_id = str(uuid4())
        self.version = version
        self.status = JobStatus.PENDING
        self.total_items = total_items
        self.completed_items = 0
        self.created_at = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
        self.results: List[Any] = []
        self.summary: Optional[BatchSummary] = None
        self.error: Optional[str] = None
        self._lock = asyncio.Lock()
    
    async def update_progress(self, completed: int):
        """Обновляет прогресс выполнения."""
        async with self._lock:
            self.completed_items = completed
            if self.status == JobStatus.PENDING:
                self.status = JobStatus.PROCESSING
    
    async def complete(self, results: List[Any], summary: BatchSummary):
        """Помечает задачу как завершённую."""
        async with self._lock:
            self.status = JobStatus.COMPLETED
            self.completed_at = datetime.utcnow()
            self.results = results
            self.summary = summary
            self.completed_items = self.total_items
    
    async def fail(self, error: str):
        """Помечает задачу как неудавшуюся."""
        async with self._lock:
            self.status = JobStatus.FAILED
            self.completed_at = datetime.utcnow()
            self.error = error
    
    def to_response(self, include_results: bool = True) -> JobInfoResponse:
        """Преобразует в модель ответа."""
        return JobInfoResponse(
            job_id=self.job_id,
            status=self.status,
            version=self.version,
            progress=JobProgress(completed=self.completed_items, total=self.total_items),
            created_at=self.created_at,
            completed_at=self.completed_at,
            results=self.results if include_results and self.status == JobStatus.COMPLETED else None,
            summary=self.summary if self.status == JobStatus.COMPLETED else None,
            error=self.error
        )


class JobStore:
    """
    Хранилище асинхронных задач.
    
    Потокобезопасная in-memory реализация с автоматической очисткой старых задач.
    """
    
    def __init__(self, retention_hours: int = 24):
        """
        Инициализирует хранилище.
        
        Args:
            retention_hours: Время хранения завершённых задач в часах
        """
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self._retention_hours = retention_hours
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start_cleanup_task(self):
        """Запускает фоновую задачу очистки."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Запущена фоновая задача очистки старых jobs")
    
    async def stop_cleanup_task(self):
        """Останавливает фоновую задачу очистки."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Остановлена фоновая задача очистки")
    
    async def _cleanup_loop(self):
        """Периодически очищает старые задачи."""
        while True:
            try:
                await asyncio.sleep(3600)  # Проверка каждый час
                await self._cleanup_old_jobs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка в cleanup loop: {e}")
    
    async def _cleanup_old_jobs(self):
        """Удаляет задачи старше retention_hours."""
        cutoff = datetime.utcnow() - timedelta(hours=self._retention_hours)
        to_delete = []
        
        async with self._lock:
            for job_id, job in self._jobs.items():
                if job.completed_at and job.completed_at < cutoff:
                    to_delete.append(job_id)
            
            for job_id in to_delete:
                del self._jobs[job_id]
        
        if to_delete:
            logger.info(f"Очищено {len(to_delete)} старых задач")
    
    async def create_job(self, version: Literal["v1", "v2"], total_items: int) -> Job:
        """
        Создаёт новую задачу.
        
        Args:
            version: Версия оценки (v1 или v2)
            total_items: Количество элементов для обработки
            
        Returns:
            Созданная задача
        """
        job = Job(version=version, total_items=total_items)
        
        async with self._lock:
            self._jobs[job.job_id] = job
        
        logger.info(f"Создана задача {job.job_id} ({version}, {total_items} элементов)")
        return job
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """
        Возвращает задачу по ID.
        
        Args:
            job_id: Идентификатор задачи
            
        Returns:
            Задача или None, если не найдена
        """
        async with self._lock:
            return self._jobs.get(job_id)
    
    async def delete_job(self, job_id: str) -> bool:
        """
        Удаляет задачу.
        
        Args:
            job_id: Идентификатор задачи
            
        Returns:
            True, если задача была удалена
        """
        async with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                logger.info(f"Удалена задача {job_id}")
                return True
            return False
    
    async def list_jobs(self, status: Optional[JobStatus] = None) -> List[Job]:
        """
        Возвращает список задач.
        
        Args:
            status: Фильтр по статусу (опционально)
            
        Returns:
            Список задач
        """
        async with self._lock:
            jobs = list(self._jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        return jobs
