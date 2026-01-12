"""
Роутер для управления асинхронными задачами.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from server.dependencies import get_job_store
from server.models import (
    ErrorResponse,
    JobDeletedResponse,
    JobInfoResponse,
    JobStatus,
)
from server.services.job_store import JobStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get(
    "",
    response_model=List[JobInfoResponse],
    summary="Список задач",
    description="Возвращает список всех задач. Можно фильтровать по статусу."
)
async def list_jobs(
    status_filter: Optional[JobStatus] = Query(None, alias="status", description="Фильтр по статусу"),
    job_store: JobStore = Depends(get_job_store)
) -> List[JobInfoResponse]:
    """Возвращает список всех задач."""
    jobs = await job_store.list_jobs(status=status_filter)
    return [job.to_response(include_results=False) for job in jobs]


@router.get(
    "/{job_id}",
    response_model=JobInfoResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Информация о задаче",
    description="Возвращает информацию о задаче, включая результаты для завершённых задач."
)
async def get_job(
    job_id: str,
    job_store: JobStore = Depends(get_job_store)
) -> JobInfoResponse:
    """Возвращает информацию о задаче по ID."""
    job = await job_store.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Задача {job_id} не найдена"
        )
    
    return job.to_response(include_results=True)


@router.delete(
    "/{job_id}",
    response_model=JobDeletedResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Удаление задачи",
    description="Удаляет задачу и её результаты из хранилища."
)
async def delete_job(
    job_id: str,
    job_store: JobStore = Depends(get_job_store)
) -> JobDeletedResponse:
    """Удаляет задачу по ID."""
    deleted = await job_store.delete_job(job_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Задача {job_id} не найдена"
        )
    
    return JobDeletedResponse(job_id=job_id, deleted=True)
