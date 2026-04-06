from fastapi import APIRouter, Request, HTTPException
from typing import Any, Dict, List

router = APIRouter(prefix="/api/results", tags=["results"])


@router.get("/{model_id}")
async def get_model_results(model_id: str, request: Request) -> Dict[str, Any]:
    """Возвращает полные данные модели: метаданные + все прогоны с результатами."""
    log_store = request.app.state.log_store
    db = request.app.state.db

    runs = log_store.get_model_runs(model_id)
    if not runs:
        raise HTTPException(status_code=404, detail="Модель не найдена")

    tags = await db.get_model_tags(model_id)

    # Формируем ответ без вложенных results (чтобы не грузить всё сразу)
    runs_summary = []
    for i, run in enumerate(runs):
        runs_summary.append({
            "run_index": i,
            "file": run["file"],
            "timestamp": run["timestamp"],
            "run_number": run["run_number"],
            "dataset": run["dataset"],
            "judge_model": run["judge_model"],
            "total_tokens": run["total_tokens"],
            "avg_tokens": run["avg_tokens"],
            "total_dialogs": run["total_dialogs"],
            "successful_dialogs": run["successful_dialogs"],
            "level_3": run["level_3"],
            "level_2": run["level_2"],
            "level_1": run["level_1"],
            "l3_per_1k": run["l3_per_1k"],
            "l2_per_1k": run["l2_per_1k"],
            "l1_per_1k": run["l1_per_1k"],
            "weighted_per_1k": run["weighted_per_1k"],
            "weighted_per_avg": run["weighted_per_avg"],
            "error_types": run["error_types"],
            "error_types_by_level": run["error_types_by_level"],
        })

    from web.backend.services.log_parser import aggregate_runs
    aggregated = aggregate_runs(runs)

    return {
        "model_id": model_id,
        "display_name": log_store.get_model_name(model_id),
        "tags": tags,
        "aggregated": aggregated,
        "runs": runs_summary,
    }


@router.get("/{model_id}/run/{run_index}")
async def get_run_results(model_id: str, run_index: int, request: Request) -> Dict[str, Any]:
    """Возвращает полные результаты одного прогона (включая все ответы)."""
    log_store = request.app.state.log_store

    runs = log_store.get_model_runs(model_id)
    if not runs:
        raise HTTPException(status_code=404, detail="Модель не найдена")

    if run_index < 0 or run_index >= len(runs):
        raise HTTPException(status_code=404, detail="Прогон не найден")

    run = runs[run_index]

    # Возвращаем результаты без тяжёлых полей (splitted_answer обрезаем)
    results = []
    for r in run["results"]:
        results.append({
            "dialog_id": r.get("dialog_id"),
            "dialog": r.get("dialog", []),
            "answer": r.get("answer", ""),
            "tokens": r.get("tokens", 0),
            "error": r.get("error"),
            "mistakes": r.get("mistakes", []),
            "mistakes_count": r.get("mistakes_count", {}),
            "splitted_answer": r.get("splitted_answer", ""),
        })

    return {
        "model_id": model_id,
        "display_name": log_store.get_model_name(model_id),
        "run_index": run_index,
        "file": run["file"],
        "timestamp": run["timestamp"],
        "dataset": run["dataset"],
        "judge_model": run["judge_model"],
        "results": results,
    }


@router.get("/{model_id}/combined")
async def get_combined_results(model_id: str, request: Request) -> Dict[str, Any]:
    """Возвращает объединённые результаты по всем прогонам (для каждого dialog_id — все прогоны)."""
    log_store = request.app.state.log_store

    runs = log_store.get_model_runs(model_id)
    if not runs:
        raise HTTPException(status_code=404, detail="Модель не найдена")

    # Группируем по dialog_id
    combined: Dict[int, List[Dict[str, Any]]] = {}
    for run_idx, run in enumerate(runs):
        for r in run["results"]:
            did = r.get("dialog_id", 0)
            if did not in combined:
                combined[did] = []
            combined[did].append({
                "run_index": run_idx,
                "answer": r.get("answer", ""),
                "tokens": r.get("tokens", 0),
                "error": r.get("error"),
                "mistakes": r.get("mistakes", []),
                "mistakes_count": r.get("mistakes_count", {}),
                "splitted_answer": r.get("splitted_answer", ""),
            })

    # Промпты берём из первого прогона
    dialogs = {}
    for r in runs[0]["results"]:
        dialogs[r.get("dialog_id", 0)] = r.get("dialog", [])

    results = []
    for did in sorted(combined.keys()):
        results.append({
            "dialog_id": did,
            "dialog": dialogs.get(did, []),
            "runs": combined[did],
        })

    return {
        "model_id": model_id,
        "display_name": log_store.get_model_name(model_id),
        "num_runs": len(runs),
        "results": results,
    }
