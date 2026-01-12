"""
Роутер для эндпоинтов оценки качества русского языка.
"""

import asyncio
import logging
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from server.dependencies import get_evaluator, get_job_store
from server.models import (
    BatchRequest,
    BatchResponseV1,
    BatchResponseV2,
    BatchSummary,
    EvaluateRequest,
    EvaluateResponseV1,
    EvaluateResponseV2,
    ErrorResponse,
    JobCreatedResponse,
    JobStatus,
)
from server.services.evaluator import EvaluatorService
from server.services.job_store import JobStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["evaluate"])


# ============================================================================
# Одиночная оценка
# ============================================================================

@router.post(
    "/v1/evaluate",
    response_model=EvaluateResponseV1,
    responses={500: {"model": ErrorResponse}},
    summary="Оценка V1 - подсчёт ошибок по категориям",
    description="Оценивает один ответ по методологии V1. Возвращает количество критических, обычных и дополнительных ошибок."
)
async def evaluate_v1(
    request: EvaluateRequest,
    evaluator: EvaluatorService = Depends(get_evaluator)
) -> EvaluateResponseV1:
    """Оценивает один ответ по методологии V1."""
    try:
        dialog = [{"role": m.role, "content": m.content} for m in request.dialog]
        result = await evaluator.evaluate_v1(dialog, request.answer)
        return result
    except Exception as e:
        logger.error(f"Ошибка при оценке V1: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при оценке: {str(e)}"
        )


@router.post(
    "/v2/evaluate",
    response_model=EvaluateResponseV2,
    responses={500: {"model": ErrorResponse}},
    summary="Оценка V2 - детальный список ошибок",
    description="Оценивает один ответ по методологии V2. Возвращает детальный список ошибок с позициями в тексте."
)
async def evaluate_v2(
    request: EvaluateRequest,
    evaluator: EvaluatorService = Depends(get_evaluator)
) -> EvaluateResponseV2:
    """Оценивает один ответ по методологии V2."""
    try:
        dialog = [{"role": m.role, "content": m.content} for m in request.dialog]
        result = await evaluator.evaluate_v2(dialog, request.answer)
        return result
    except Exception as e:
        logger.error(f"Ошибка при оценке V2: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при оценке: {str(e)}"
        )


# ============================================================================
# Синхронная батчевая оценка
# ============================================================================

@router.post(
    "/v1/evaluate/batch",
    response_model=BatchResponseV1,
    responses={500: {"model": ErrorResponse}},
    summary="Синхронная батчевая оценка V1",
    description="Оценивает несколько ответов по методологии V1. Клиент ждёт завершения всех оценок."
)
async def evaluate_batch_v1(
    request: BatchRequest,
    evaluator: EvaluatorService = Depends(get_evaluator)
) -> BatchResponseV1:
    """Оценивает батч элементов по методологии V1 синхронно."""
    try:
        results, summary = await evaluator.evaluate_batch_v1(request.items)
        return BatchResponseV1(results=results, summary=summary)
    except Exception as e:
        logger.error(f"Ошибка при батчевой оценке V1: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при батчевой оценке: {str(e)}"
        )


@router.post(
    "/v2/evaluate/batch",
    response_model=BatchResponseV2,
    responses={500: {"model": ErrorResponse}},
    summary="Синхронная батчевая оценка V2",
    description="Оценивает несколько ответов по методологии V2. Клиент ждёт завершения всех оценок."
)
async def evaluate_batch_v2(
    request: BatchRequest,
    evaluator: EvaluatorService = Depends(get_evaluator)
) -> BatchResponseV2:
    """Оценивает батч элементов по методологии V2 синхронно."""
    try:
        results, summary = await evaluator.evaluate_batch_v2(request.items)
        return BatchResponseV2(results=results, summary=summary)
    except Exception as e:
        logger.error(f"Ошибка при батчевой оценке V2: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при батчевой оценке: {str(e)}"
        )


# ============================================================================
# Асинхронная батчевая оценка
# ============================================================================

async def _process_batch_async_v1(
    job_id: str,
    request: BatchRequest,
    evaluator: EvaluatorService,
    job_store: JobStore
):
    """Фоновая задача для асинхронной обработки батча V1."""
    job = await job_store.get_job(job_id)
    if not job:
        logger.error(f"Job {job_id} не найден")
        return
    
    completed = 0
    
    async def progress_callback():
        nonlocal completed
        completed += 1
        await job.update_progress(completed)
    
    try:
        results, summary = await evaluator.evaluate_batch_v1(request.items, progress_callback)
        await job.complete(results, summary)
        logger.info(f"Job {job_id} завершён успешно")
    except Exception as e:
        logger.error(f"Job {job_id} завершился с ошибкой: {e}", exc_info=True)
        await job.fail(str(e))


async def _process_batch_async_v2(
    job_id: str,
    request: BatchRequest,
    evaluator: EvaluatorService,
    job_store: JobStore
):
    """Фоновая задача для асинхронной обработки батча V2."""
    job = await job_store.get_job(job_id)
    if not job:
        logger.error(f"Job {job_id} не найден")
        return
    
    completed = 0
    
    async def progress_callback():
        nonlocal completed
        completed += 1
        await job.update_progress(completed)
    
    try:
        results, summary = await evaluator.evaluate_batch_v2(request.items, progress_callback)
        await job.complete(results, summary)
        logger.info(f"Job {job_id} завершён успешно")
    except Exception as e:
        logger.error(f"Job {job_id} завершился с ошибкой: {e}", exc_info=True)
        await job.fail(str(e))


@router.post(
    "/v1/evaluate/batch/async",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={500: {"model": ErrorResponse}},
    summary="Асинхронная батчевая оценка V1",
    description="Создаёт асинхронную задачу для обработки батча по методологии V1. Возвращает job_id для отслеживания прогресса."
)
async def evaluate_batch_async_v1(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    evaluator: EvaluatorService = Depends(get_evaluator),
    job_store: JobStore = Depends(get_job_store)
) -> JobCreatedResponse:
    """Создаёт асинхронную задачу для батчевой оценки V1."""
    try:
        job = await job_store.create_job(version="v1", total_items=len(request.items))
        
        # Запускаем фоновую задачу
        background_tasks.add_task(
            _process_batch_async_v1,
            job.job_id,
            request,
            evaluator,
            job_store
        )
        
        return JobCreatedResponse(
            job_id=job.job_id,
            status=JobStatus.PENDING,
            created_at=job.created_at,
            total_items=len(request.items)
        )
    except Exception as e:
        logger.error(f"Ошибка при создании асинхронной задачи V1: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при создании задачи: {str(e)}"
        )


@router.post(
    "/v2/evaluate/batch/async",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={500: {"model": ErrorResponse}},
    summary="Асинхронная батчевая оценка V2",
    description="Создаёт асинхронную задачу для обработки батча по методологии V2. Возвращает job_id для отслеживания прогресса."
)
async def evaluate_batch_async_v2(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    evaluator: EvaluatorService = Depends(get_evaluator),
    job_store: JobStore = Depends(get_job_store)
) -> JobCreatedResponse:
    """Создаёт асинхронную задачу для батчевой оценки V2."""
    try:
        job = await job_store.create_job(version="v2", total_items=len(request.items))
        
        # Запускаем фоновую задачу
        background_tasks.add_task(
            _process_batch_async_v2,
            job.job_id,
            request,
            evaluator,
            job_store
        )
        
        return JobCreatedResponse(
            job_id=job.job_id,
            status=JobStatus.PENDING,
            created_at=job.created_at,
            total_items=len(request.items)
        )
    except Exception as e:
        logger.error(f"Ошибка при создании асинхронной задачи V2: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при создании задачи: {str(e)}"
        )
