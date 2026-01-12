"""
Pydantic модели для запросов и ответов API.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Входные модели
# ============================================================================

class Message(BaseModel):
    """Сообщение в диалоге."""
    role: Literal["user", "assistant", "system"]
    content: str


class EvaluateRequest(BaseModel):
    """Запрос на оценку одного ответа."""
    dialog: List[Message] = Field(..., description="История диалога (включая последний запрос пользователя)")
    answer: str = Field(..., description="Ответ модели для оценки")


class BatchItem(BaseModel):
    """Элемент батча для оценки."""
    id: str = Field(..., description="Уникальный идентификатор элемента")
    dialog: List[Message] = Field(..., description="История диалога")
    answer: str = Field(..., description="Ответ модели для оценки")


class BatchRequest(BaseModel):
    """Запрос на батчевую оценку."""
    items: List[BatchItem] = Field(..., description="Список элементов для оценки")


# ============================================================================
# Выходные модели V1
# ============================================================================

class EvaluateResponseV1(BaseModel):
    """Ответ на оценку V1 - подсчёт ошибок по категориям."""
    critical_mistakes: int = Field(..., description="Количество критических ошибок")
    mistakes: int = Field(..., description="Количество обычных ошибок")
    additional_mistakes: int = Field(..., description="Количество дополнительных ошибок")
    explanation_critical_mistakes: List[str] = Field(default_factory=list, description="Описания критических ошибок")
    explanation_mistakes: List[str] = Field(default_factory=list, description="Описания обычных ошибок")
    explanation_additional_mistakes: List[str] = Field(default_factory=list, description="Описания дополнительных ошибок")


class BatchItemResultV1(EvaluateResponseV1):
    """Результат оценки одного элемента батча V1."""
    id: str = Field(..., description="Идентификатор элемента")
    error: Optional[str] = Field(None, description="Сообщение об ошибке, если оценка не удалась")


class BatchSummary(BaseModel):
    """Сводка по результатам батчевой обработки."""
    total: int = Field(..., description="Всего элементов")
    successful: int = Field(..., description="Успешно обработано")
    failed: int = Field(..., description="Ошибок при обработке")


class BatchResponseV1(BaseModel):
    """Ответ на батчевую оценку V1."""
    results: List[BatchItemResultV1] = Field(..., description="Результаты оценки")
    summary: BatchSummary = Field(..., description="Сводка")


# ============================================================================
# Выходные модели V2
# ============================================================================

class MistakeV2(BaseModel):
    """Описание ошибки в формате V2."""
    position: List[int] = Field(..., description="Позиции предложений с ошибкой")
    level: int = Field(..., description="Уровень ошибки (1-3)")
    type: str = Field(..., description="Тип ошибки")
    explanation: str = Field(..., description="Описание ошибки")


class MistakesCount(BaseModel):
    """Подсчёт ошибок по уровням."""
    level_1: int = Field(0, alias="1", description="Незначительные ошибки")
    level_2: int = Field(0, alias="2", description="Обычные ошибки")
    level_3: int = Field(0, alias="3", description="Критические ошибки")
    
    class Config:
        populate_by_name = True


class EvaluateResponseV2(BaseModel):
    """Ответ на оценку V2 - детальный список ошибок."""
    mistakes: List[MistakeV2] = Field(default_factory=list, description="Список найденных ошибок")
    mistakes_count: Dict[str, int] = Field(default_factory=lambda: {"1": 0, "2": 0, "3": 0}, description="Подсчёт ошибок по уровням")
    splitted_answer: str = Field("", description="Ответ, разбитый на пронумерованные предложения")


class BatchItemResultV2(EvaluateResponseV2):
    """Результат оценки одного элемента батча V2."""
    id: str = Field(..., description="Идентификатор элемента")
    error: Optional[str] = Field(None, description="Сообщение об ошибке, если оценка не удалась")


class BatchResponseV2(BaseModel):
    """Ответ на батчевую оценку V2."""
    results: List[BatchItemResultV2] = Field(..., description="Результаты оценки")
    summary: BatchSummary = Field(..., description="Сводка")


# ============================================================================
# Модели для асинхронных задач
# ============================================================================

class JobStatus(str, Enum):
    """Статус асинхронной задачи."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobProgress(BaseModel):
    """Прогресс выполнения задачи."""
    completed: int = Field(0, description="Обработано элементов")
    total: int = Field(0, description="Всего элементов")


class JobCreatedResponse(BaseModel):
    """Ответ при создании асинхронной задачи."""
    job_id: str = Field(..., description="Уникальный идентификатор задачи")
    status: JobStatus = Field(JobStatus.PENDING, description="Статус задачи")
    created_at: datetime = Field(..., description="Время создания")
    total_items: int = Field(..., description="Количество элементов для обработки")


class JobInfoResponse(BaseModel):
    """Информация о задаче."""
    job_id: str = Field(..., description="Идентификатор задачи")
    status: JobStatus = Field(..., description="Статус задачи")
    version: Literal["v1", "v2"] = Field(..., description="Версия оценки")
    progress: JobProgress = Field(..., description="Прогресс выполнения")
    created_at: datetime = Field(..., description="Время создания")
    completed_at: Optional[datetime] = Field(None, description="Время завершения")
    results: Optional[List[Any]] = Field(None, description="Результаты (только для завершённых задач)")
    summary: Optional[BatchSummary] = Field(None, description="Сводка (только для завершённых задач)")
    error: Optional[str] = Field(None, description="Сообщение об ошибке")


class JobDeletedResponse(BaseModel):
    """Ответ при удалении задачи."""
    job_id: str = Field(..., description="Идентификатор удалённой задачи")
    deleted: bool = Field(True, description="Флаг успешного удаления")


# ============================================================================
# Общие модели
# ============================================================================

class ErrorResponse(BaseModel):
    """Ответ с ошибкой."""
    detail: str = Field(..., description="Описание ошибки")
