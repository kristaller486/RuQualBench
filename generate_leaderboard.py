import json
import statistics
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime
from jinja2 import Environment, FileSystemLoader


def parse_logs(logs_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Парсит все JSON логи и группирует по моделям"""
    models_data = defaultdict(list)
    
    for log_file in logs_dir.glob("*.json"):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            config = data.get("config", {})
            summary = data.get("summary", {})
            
            model_name = config.get("model", "Unknown")
            verbose_name = config.get("verbose_name")
            display_name = verbose_name if verbose_name else model_name
            
            critical = summary.get("critical_mistakes_per_1000_tokens", 0)
            mistakes = summary.get("mistakes_per_1000_tokens", 0)
            additional = summary.get("additional_mistakes_per_1000_tokens", 0)
            normalized_total = critical * 2 + mistakes + additional * 0.5

            models_data[display_name].append({
                "critical": critical,
                "mistakes": mistakes,
                "additional": additional,
                "total": normalized_total,
                "tokens": summary.get("total_tokens", 0),
                "file": log_file.name
            })
        except Exception as e:
            print(f"Ошибка при парсинге {log_file}: {e}")
            continue
    
    return models_data


def aggregate_model_data(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Агрегирует данные для модели с несколькими прогонами"""
    if len(runs) == 1:
        return {
            **runs[0],
            "num_runs": 1,
            "critical_se": None,
            "mistakes_se": None,
            "additional_se": None,
            "total_se": None
        }
    
    critical_values = [r["critical"] for r in runs]
    mistakes_values = [r["mistakes"] for r in runs]
    additional_values = [r["additional"] for r in runs]
    total_values = [r["total"] for r in runs]
    tokens_values = [r["tokens"] for r in runs]
    
    def calc_se(values):
        if len(values) < 2:
            return 0
        stdev = statistics.stdev(values)
        return stdev / (len(values) ** 0.5)
    
    return {
        "critical": round(statistics.mean(critical_values), 2),
        "critical_se": round(calc_se(critical_values), 2) if len(runs) >= 2 else None,
        "mistakes": round(statistics.mean(mistakes_values), 2),
        "mistakes_se": round(calc_se(mistakes_values), 2) if len(runs) >= 2 else None,
        "additional": round(statistics.mean(additional_values), 2),
        "additional_se": round(calc_se(additional_values), 2) if len(runs) >= 2 else None,
        "total": round(statistics.mean(total_values), 2),
        "total_se": round(calc_se(total_values), 2) if len(runs) >= 2 else None,
        "tokens": int(statistics.mean(tokens_values)),
        "num_runs": len(runs),
        "file": runs[0]["file"]
    }


def generate_leaderboard():
    """Генерирует HTML лидерборд из JSON логов"""
    logs_dir = Path("logs")
    
    if not logs_dir.exists():
        print("Папка logs/ не найдена!")
        return
    
    # Парсим и группируем данные
    print("Парсинг JSON логов...")
    models_data = parse_logs(logs_dir)
    
    if not models_data:
        print("Не найдено валидных JSON логов!")
        return
    
    # Агрегируем данные для каждой модели
    print("Агрегация данных...")
    leaderboard = []
    for model_name, runs in models_data.items():
        aggregated = aggregate_model_data(runs)
        leaderboard.append({
            "model": model_name,
            **aggregated
        })
    
    # Сортируем по общему количеству ошибок (меньше = лучше)
    leaderboard.sort(key=lambda x: x["total"])
    
    # Находим максимальный score для прогресс-бара
    max_score = max([entry["total"] for entry in leaderboard]) if leaderboard else 100
    
    # Рендерим шаблон
    print("Генерация HTML...")
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('leaderboard.jinja2')
    
    html_content = template.render(
        leaderboard=leaderboard,
        max_score=max_score,
        update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    # Сохраняем
    output_file = Path("index.html")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Лидерборд сгенерирован: {output_file}")
    print(f"📊 Моделей в лидерборде: {len(leaderboard)}")
    
    # Выводим топ-3
    print("\n🏆 ТОП-3:")
    for i, entry in enumerate(leaderboard[:3], 1):
        se_str = f" ± {entry['total_se']:.2f}" if entry.get('total_se') else ""
        print(f"  {i}. {entry['model']}: {entry['total']:.2f}{se_str} нормировано ошибок/1000 токенов")


if __name__ == "__main__":
    generate_leaderboard()
