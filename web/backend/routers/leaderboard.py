from fastapi import APIRouter, Request
from typing import Any, Dict, List

router = APIRouter(prefix="/api/leaderboard", tags=["leaderboard"])


@router.get("")
async def get_leaderboard(request: Request) -> List[Dict[str, Any]]:
    """Возвращает данные лидерборда с учётом enabled/disabled из БД."""
    log_store = request.app.state.log_store
    db = request.app.state.db

    leaderboard = log_store.get_leaderboard()
    all_tags = await db.get_all_tags_by_models()

    result = []
    rank = 0
    for entry in leaderboard:
        mid = entry["model_id"]
        enabled = await db.is_model_enabled(mid)
        if not enabled:
            continue
        rank += 1
        result.append({
            **entry,
            "rank": rank,
            "tags": all_tags.get(mid, {}),
        })

    return result


@router.get("/all")
async def get_leaderboard_all(request: Request) -> List[Dict[str, Any]]:
    """Возвращает все модели (включая выключенные) — для админки."""
    log_store = request.app.state.log_store
    db = request.app.state.db

    leaderboard = log_store.get_leaderboard()
    all_tags = await db.get_all_tags_by_models()

    result = []
    rank = 0
    for entry in leaderboard:
        mid = entry["model_id"]
        enabled = await db.is_model_enabled(mid)
        if enabled:
            rank += 1
        result.append({
            **entry,
            "rank": rank if enabled else None,
            "enabled": enabled,
            "tags": all_tags.get(mid, {}),
        })

    return result
