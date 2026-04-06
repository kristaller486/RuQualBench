from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional

router = APIRouter(prefix="/api/admin", tags=["admin"])


class ToggleModelRequest(BaseModel):
    enabled: bool


class UpdateTagsRequest(BaseModel):
    tags: Dict[str, str]


class TagTypeRequest(BaseModel):
    key: str
    label: str
    description: str = ""


@router.post("/models/{model_id}/toggle")
async def toggle_model(model_id: str, body: ToggleModelRequest, request: Request):
    """Включает/выключает модель в лидерборде."""
    db = request.app.state.db
    model = await db.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Модель не найдена в БД")
    await db.set_model_enabled(model_id, body.enabled)
    return {"ok": True, "model_id": model_id, "enabled": body.enabled}


@router.post("/models/{model_id}/tags")
async def update_model_tags(model_id: str, body: UpdateTagsRequest, request: Request):
    """Обновляет теги (метаданные) модели."""
    db = request.app.state.db
    model = await db.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Модель не найдена в БД")
    await db.set_model_tags(model_id, body.tags)
    return {"ok": True, "model_id": model_id, "tags": body.tags}


@router.get("/models")
async def list_models(request: Request):
    """Список всех моделей из БД."""
    db = request.app.state.db
    return await db.get_all_models()


@router.get("/tag-types")
async def get_tag_types(request: Request):
    """Список доступных типов тегов."""
    db = request.app.state.db
    return await db.get_tag_types()


@router.post("/tag-types")
async def add_tag_type(body: TagTypeRequest, request: Request):
    """Добавляет или обновляет тип тега."""
    db = request.app.state.db
    await db.add_tag_type(body.key, body.label, body.description)
    return {"ok": True}


@router.delete("/tag-types/{key}")
async def delete_tag_type(key: str, request: Request):
    """Удаляет тип тега."""
    db = request.app.state.db
    await db.delete_tag_type(key)
    return {"ok": True}


@router.post("/sync")
async def sync_from_github(request: Request):
    """Выполняет git pull и перезагружает логи."""
    sync_service = request.app.state.sync_service
    log_store = request.app.state.log_store
    db = request.app.state.db

    output = await sync_service.pull()
    log_store.reload()

    # Обновляем записи моделей в БД
    for mid in log_store.get_model_ids():
        name = log_store.get_model_name(mid)
        await db.ensure_model(mid, name)

    return {"ok": True, "output": output}


@router.get("/status")
async def get_status(request: Request):
    """Статус последней синхронизации и общая информация."""
    sync_service = request.app.state.sync_service
    log_store = request.app.state.log_store

    return {
        "sync": sync_service.get_status(),
        "total_models": len(log_store.get_model_ids()),
        "leaderboard_entries": len(log_store.get_leaderboard()),
    }
