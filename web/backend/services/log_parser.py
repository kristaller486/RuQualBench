import json
import hashlib
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict


# Типы ошибок V2
ERROR_TYPES = [
    "incorrect_agreement",
    "other_language_insert",
    "syntax",
    "calque",
    "made_up_words",
    "wrong_capitalization",
    "tautology",
    "grammatical_gender_change",
    "other",
]

def _model_id(config: Dict[str, Any]) -> str:
    """Генерирует стабильный ID модели из конфига."""
    key = config.get("verbose_name") or config.get("model", "unknown")
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _display_name(config: Dict[str, Any]) -> str:
    return config.get("verbose_name") or config.get("model", "Unknown")


def _calc_se(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    return statistics.stdev(values) / (len(values) ** 0.5)


def _get_mistakes_list(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Извлекает список ошибок из результата, поддерживая оба формата (V1 и V2)."""
    mistakes = result.get("mistakes", [])
    if isinstance(mistakes, list):
        return mistakes
    return []


def _is_v2_format(results: List[Dict[str, Any]]) -> bool:
    """Проверяет, является ли результат V2-форматом (mistakes — список объектов)."""
    for r in results:
        m = r.get("mistakes")
        if isinstance(m, list) and len(m) > 0 and isinstance(m[0], dict):
            return True
        if isinstance(m, int):
            return False
    return True


def _count_errors_by_type(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Подсчитывает количество ошибок каждого типа по всем результатам."""
    counts: Dict[str, int] = {t: 0 for t in ERROR_TYPES}
    for r in results:
        for m in _get_mistakes_list(r):
            err_type = m.get("type", "other")
            if err_type in counts:
                counts[err_type] += 1
            else:
                counts["other"] += 1
    return counts


def _count_errors_by_type_and_level(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Подсчитывает ошибки по типам и уровням."""
    counts: Dict[str, Dict[str, int]] = {}
    for t in ERROR_TYPES:
        counts[t] = {"1": 0, "2": 0, "3": 0}
    for r in results:
        for m in _get_mistakes_list(r):
            err_type = m.get("type", "other")
            level = str(m.get("level", 1))
            if err_type not in counts:
                err_type = "other"
            if level in counts[err_type]:
                counts[err_type][level] += 1
    return counts


def _is_v2_log(results: List[Dict[str, Any]]) -> bool:
    """Проверяет, что лог в V2-формате (mistakes — список объектов с position/level/type)."""
    for r in results:
        m = r.get("mistakes")
        if isinstance(m, list) and len(m) > 0:
            return isinstance(m[0], dict) and "position" in m[0]
        if isinstance(m, int):
            return False
    return True


def parse_single_log(filepath: Path) -> Optional[Dict[str, Any]]:
    """Парсит один JSON лог-файл V2. Пропускает файлы в V1-формате."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    config = data.get("config", {})
    summary = data.get("summary", {})
    results = data.get("results", [])

    if not config or not results:
        return None

    if not _is_v2_log(results):
        return None

    total_tokens = summary.get("total_tokens", 0)
    if total_tokens == 0:
        total_tokens = sum(r.get("tokens", 0) for r in results if r.get("error") is None)

    valid_results = [r for r in results if r.get("error") is None]

    level_3 = sum(r.get("mistakes_count", {}).get("3", 0) for r in valid_results)
    level_2 = sum(r.get("mistakes_count", {}).get("2", 0) for r in valid_results)
    level_1 = sum(r.get("mistakes_count", {}).get("1", 0) for r in valid_results)

    l3_per_1k = (level_3 / total_tokens * 1000) if total_tokens > 0 else 0
    l2_per_1k = (level_2 / total_tokens * 1000) if total_tokens > 0 else 0
    l1_per_1k = (level_1 / total_tokens * 1000) if total_tokens > 0 else 0
    weighted_per_1k = l3_per_1k * 2 + l2_per_1k + l1_per_1k * 0.5

    avg_tokens = total_tokens / len(valid_results) if valid_results else 0
    weighted_per_avg = weighted_per_1k * avg_tokens / 1000

    error_types = _count_errors_by_type(valid_results)
    error_types_by_level = _count_errors_by_type_and_level(valid_results)

    return {
        "model_id": _model_id(config),
        "display_name": _display_name(config),
        "original_model": config.get("model", ""),
        "judge_model": config.get("judge_model", ""),
        "dataset": config.get("dataset", ""),
        "timestamp": config.get("timestamp", ""),
        "run_number": config.get("run_number", 1),
        "total_runs": config.get("total_runs", 1),
        "file": filepath.name,
        "total_tokens": total_tokens,
        "avg_tokens": round(avg_tokens, 1),
        "total_dialogs": len(results),
        "successful_dialogs": len(valid_results),
        "level_3": level_3,
        "level_2": level_2,
        "level_1": level_1,
        "l3_per_1k": round(l3_per_1k, 4),
        "l2_per_1k": round(l2_per_1k, 4),
        "l1_per_1k": round(l1_per_1k, 4),
        "weighted_per_1k": round(weighted_per_1k, 4),
        "weighted_per_avg": round(weighted_per_avg, 4),
        "error_types": error_types,
        "error_types_by_level": error_types_by_level,
        "results": results,
    }


def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Агрегирует несколько прогонов одной модели."""
    if len(runs) == 1:
        r = runs[0]
        return {
            "model_id": r["model_id"],
            "display_name": r["display_name"],
            "original_model": r["original_model"],
            "judge_model": r["judge_model"],
            "dataset": r["dataset"],
            "num_runs": 1,
            "l3_per_1k": r["l3_per_1k"],
            "l3_per_1k_se": None,
            "l2_per_1k": r["l2_per_1k"],
            "l2_per_1k_se": None,
            "l1_per_1k": r["l1_per_1k"],
            "l1_per_1k_se": None,
            "weighted_per_1k": r["weighted_per_1k"],
            "weighted_per_1k_se": None,
            "weighted_per_avg": r["weighted_per_avg"],
            "weighted_per_avg_se": None,
            "avg_tokens": r["avg_tokens"],
            "total_tokens": r["total_tokens"],
            "error_types": r["error_types"],
            "error_types_by_level": r["error_types_by_level"],
            "files": [r["file"]],
        }

    return {
        "model_id": runs[0]["model_id"],
        "display_name": runs[0]["display_name"],
        "original_model": runs[0]["original_model"],
        "judge_model": runs[0]["judge_model"],
        "dataset": runs[0]["dataset"],
        "num_runs": len(runs),
        "l3_per_1k": round(statistics.mean([r["l3_per_1k"] for r in runs]), 4),
        "l3_per_1k_se": round(_calc_se([r["l3_per_1k"] for r in runs]) or 0, 4),
        "l2_per_1k": round(statistics.mean([r["l2_per_1k"] for r in runs]), 4),
        "l2_per_1k_se": round(_calc_se([r["l2_per_1k"] for r in runs]) or 0, 4),
        "l1_per_1k": round(statistics.mean([r["l1_per_1k"] for r in runs]), 4),
        "l1_per_1k_se": round(_calc_se([r["l1_per_1k"] for r in runs]) or 0, 4),
        "weighted_per_1k": round(statistics.mean([r["weighted_per_1k"] for r in runs]), 4),
        "weighted_per_1k_se": round(_calc_se([r["weighted_per_1k"] for r in runs]) or 0, 4),
        "weighted_per_avg": round(statistics.mean([r["weighted_per_avg"] for r in runs]), 4),
        "weighted_per_avg_se": round(_calc_se([r["weighted_per_avg"] for r in runs]) or 0, 4),
        "avg_tokens": round(statistics.mean([r["avg_tokens"] for r in runs]), 1),
        "total_tokens": sum(r["total_tokens"] for r in runs),
        "error_types": _merge_error_types([r["error_types"] for r in runs]),
        "error_types_by_level": _merge_error_types_by_level([r["error_types_by_level"] for r in runs]),
        "files": [r["file"] for r in runs],
    }


def _merge_error_types(type_dicts: List[Dict[str, int]]) -> Dict[str, int]:
    """Суммирует ошибки по типам из нескольких прогонов."""
    merged: Dict[str, int] = {t: 0 for t in ERROR_TYPES}
    for d in type_dicts:
        for k, v in d.items():
            merged[k] = merged.get(k, 0) + v
    return merged


def _merge_error_types_by_level(dicts: List[Dict[str, Dict[str, int]]]) -> Dict[str, Dict[str, int]]:
    """Суммирует ошибки по типам и уровням."""
    merged: Dict[str, Dict[str, int]] = {}
    for t in ERROR_TYPES:
        merged[t] = {"1": 0, "2": 0, "3": 0}
    for d in dicts:
        for t, levels in d.items():
            if t not in merged:
                t = "other"
            for level, count in levels.items():
                merged[t][level] = merged[t].get(level, 0) + count
    return merged


class LogStore:
    """Хранилище распарсенных логов V2."""

    def __init__(self, logs_dir: Path):
        self.logs_dir = logs_dir
        # model_id -> list[parsed_run]
        self._runs: Dict[str, List[Dict[str, Any]]] = {}
        # model_id -> aggregated
        self._leaderboard: List[Dict[str, Any]] = []
        # model_id -> display_name
        self._model_names: Dict[str, str] = {}

    def reload(self):
        """Перечитывает все логи из директории."""
        runs_by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for filepath in sorted(self.logs_dir.glob("*.json")):
            parsed = parse_single_log(filepath)
            if parsed:
                mid = parsed["model_id"]
                runs_by_model[mid].append(parsed)
                self._model_names[mid] = parsed["display_name"]

        self._runs = dict(runs_by_model)

        leaderboard = []
        for mid, runs in self._runs.items():
            agg = aggregate_runs(runs)
            leaderboard.append(agg)

        leaderboard.sort(key=lambda x: x["weighted_per_avg"])
        self._leaderboard = leaderboard

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        return self._leaderboard

    def get_model_ids(self) -> List[str]:
        return list(self._runs.keys())

    def get_model_runs(self, model_id: str) -> List[Dict[str, Any]]:
        return self._runs.get(model_id, [])

    def get_model_name(self, model_id: str) -> str:
        return self._model_names.get(model_id, "Unknown")
