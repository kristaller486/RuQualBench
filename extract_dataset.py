import argparse
import json
import logging
import re
import statistics
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

from datasets import Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def detect_generation_cycle(text: str) -> bool:
    """Проверяет наличие циклов в тексте через повторяющиеся последовательности"""
    if not text:
        return False
    
    # Регулярка для поиска повторяющихся последовательностей от 20 до 200 символов
    pattern = r'(?![\s.*|+─=\\-]{20,})(.{20,200}?)\1'
    matches = list(re.finditer(
            pattern,
            text,
            re.DOTALL
        )
    )
    
    return len(matches) > 5


def load_benchmark_logs(logs_dir: Path, filter_timestamp: str = None,
                       filter_model: str = None, filter_dataset: str = None) -> List[Dict[str, Any]]:
    """Загружает все файлы логов из директории с опциональной фильтрацией"""
    log_files = list(logs_dir.glob("benchmark_*.json"))
    
    if not log_files:
        raise ValueError(f"Не найдено файлов логов в {logs_dir}")
    
    logger.info(f"Найдено {len(log_files)} файлов логов")
    
    all_data = []
    for log_file in log_files:
        # Фильтрация по имени файла
        if filter_timestamp and filter_timestamp not in log_file.name:
            continue
        if filter_dataset and not log_file.name.endswith(f"{filter_dataset}.json"):
            continue
            
        with open(log_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Фильтрация по модели
        if filter_model and data.get("config", {}).get("model") != filter_model:
            continue
            
        all_data.append({
            "file": log_file.name,
            "data": data
        })
    
    logger.info(f"После фильтрации осталось {len(all_data)} файлов")
    return all_data


def extract_dataset_from_logs(log_data_list: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Извлекает данные из логов и формирует плоский датасет"""
    
    dataset_dict = defaultdict(list)
    seen_answers = {}  # Для отслеживания дубликатов
    
    for log_item in log_data_list:
        log_file = log_item["file"]
        data = log_item["data"]
        
        config = data.get("config", {})
        results = data.get("results", [])
        
        # Извлекаем метаданные
        test_model_name = config.get("model", "unknown")
        verbose_test_model_name = config.get("verbose_name", None)
        judge_model_name = config.get("judge_model", "unknown")
        dataset_name = config.get("dataset", "unknown")
        timestamp = config.get("timestamp", "unknown")
        run_number = config.get("run_number", 0)
        
        logger.info(f"Обработка {log_file}: {len(results)} записей")
        
        for result in results:
            dialog_id = result.get("dialog_id")
            dialog = result.get("dialog", [])
            test_answer = result.get("answer")
            tokens = result.get("tokens", 0)
            
            critical_mistakes = result.get("critical_mistakes", 0)
            mistakes = result.get("mistakes", 0)
            additional_mistakes = result.get("additional_mistakes", 0)
            
            explanation_critical_mistakes = result.get("explanation_critical_mistakes", [])
            explanation_mistakes = result.get("explanation_mistakes", [])
            explanation_additional_mistakes = result.get("explanation_additional_mistakes", [])
            
            error = result.get("error")
            has_error = error is not None
            error_message = str(error) if error else None
            
            # Проверка на дубликат answer
            is_duplicate = False
            if test_answer:
                if test_answer in seen_answers:
                    is_duplicate = True
                else:
                    seen_answers[test_answer] = True
            
            # Проверка на циклы генерации
            is_generation_cycle = detect_generation_cycle(test_answer) if test_answer else False
            
            # Добавляем запись в датасет
            dataset_dict["dialog_id"].append(dialog_id)
            dataset_dict["dialog"].append(dialog)
            dataset_dict["test_answer"].append(test_answer)
            dataset_dict["tokens"].append(tokens)
            dataset_dict["critical_mistakes"].append(critical_mistakes)
            dataset_dict["mistakes"].append(mistakes)
            dataset_dict["additional_mistakes"].append(additional_mistakes)
            dataset_dict["explanation_critical_mistakes"].append(explanation_critical_mistakes)
            dataset_dict["explanation_mistakes"].append(explanation_mistakes)
            dataset_dict["explanation_additional_mistakes"].append(explanation_additional_mistakes)
            dataset_dict["has_error"].append(has_error)
            dataset_dict["error_message"].append(error_message)
            dataset_dict["is_duplicate_test_answer"].append(is_duplicate)
            dataset_dict["is_generation_cycle"].append(is_generation_cycle)
            dataset_dict["test_model_name"].append(test_model_name)
            dataset_dict["verbose_test_model_name"].append(verbose_test_model_name)
            dataset_dict["judge_model_name"].append(judge_model_name)
            dataset_dict["dataset_name"].append(dataset_name)
            dataset_dict["timestamp"].append(timestamp)
            dataset_dict["run_number"].append(run_number)
    
    total_records = len(dataset_dict["dialog_id"])
    duplicates_count = sum(dataset_dict["is_duplicate_test_answer"])
    errors_count = sum(dataset_dict["has_error"])
    cycles_count = sum(dataset_dict["is_generation_cycle"])
    
    total_tokens = sum(dataset_dict["tokens"])
    avg_tokens = statistics.mean(dataset_dict["tokens"]) if dataset_dict["tokens"] else 0
    median_tokens = statistics.median(dataset_dict["tokens"]) if dataset_dict["tokens"] else 0
    
    logger.info(f"Всего записей: {total_records}")
    logger.info(f"Дубликатов ответов: {duplicates_count}")
    logger.info(f"Записей с ошибками: {errors_count}")
    logger.info(f"Записей с циклами генерации: {cycles_count}")
    logger.info(f"Всего токенов: {total_tokens}")
    logger.info(f"Токенов на ответ (сред): {avg_tokens:.2f}")
    logger.info(f"Токенов на ответ (медиан): {median_tokens:.2f}")
    
    return dict(dataset_dict)


def save_dataset(dataset_dict: Dict[str, List[Any]], output_path: Path):
    """Сохраняет датасет в формате Hugging Face datasets"""
    
    dataset = Dataset.from_dict(dataset_dict)
    
    logger.info(f"Создан датасет с {len(dataset)} записями")
    logger.info(f"Колонки: {dataset.column_names}")
    
    # Определяем формат по расширению
    if output_path.suffix == '.parquet':
        dataset.to_parquet(str(output_path))
        logger.info(f"Датасет сохранен в {output_path} (формат: parquet)")
    elif output_path.suffix == '.arrow':
        dataset.save_to_disk(str(output_path))
        logger.info(f"Датасет сохранен в {output_path} (формат: arrow)")
    elif output_path.suffix == '.jsonl':
        dataset.to_json(str(output_path), force_ascii=False)
        logger.info(f"Датасет сохранен в {output_path} (формат: jsonl)")
    else:
        # По умолчанию сохраняем как arrow
        dataset.save_to_disk(str(output_path))
        logger.info(f"Датасет сохранен в {output_path} (формат: arrow)")


def main():
    parser = argparse.ArgumentParser(
        description="Извлекает датасет из файлов логов бенчмарка"
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Директория с файлами логов (по умолчанию: logs)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="extracted_dataset.parquet",
        help="Путь для сохранения датасета (по умолчанию: extracted_dataset.parquet)"
    )
    parser.add_argument(
        "--filter-timestamp",
        type=str,
        help="Фильтровать по timestamp (например: 2025-10-17_15-17-05)"
    )
    parser.add_argument(
        "--filter-model",
        type=str,
        help="Фильтровать по имени модели"
    )
    parser.add_argument(
        "--filter-dataset",
        type=str,
        choices=["lite", "base", "large"],
        help="Фильтровать по типу датасета"
    )
    
    args = parser.parse_args()
    
    logs_dir = Path(args.logs_dir)
    output_path = Path(args.output)
    
    if not logs_dir.exists():
        logger.error(f"Директория {logs_dir} не существует")
        return
    
    logger.info("Начало извлечения датасета")
    logger.info(f"Директория логов: {logs_dir}")
    logger.info(f"Выходной файл: {output_path}")
    
    if args.filter_timestamp:
        logger.info(f"Фильтр timestamp: {args.filter_timestamp}")
    if args.filter_model:
        logger.info(f"Фильтр модели: {args.filter_model}")
    if args.filter_dataset:
        logger.info(f"Фильтр датасета: {args.filter_dataset}")
    
    # Загрузка логов
    log_data_list = load_benchmark_logs(
        logs_dir, 
        args.filter_timestamp, 
        args.filter_model, 
        args.filter_dataset
    )
    
    if not log_data_list:
        logger.error("Не найдено файлов логов после фильтрации")
        return
    
    # Извлечение данных
    dataset_dict = extract_dataset_from_logs(log_data_list)
    
    # Сохранение датасета
    save_dataset(dataset_dict, output_path)
    
    logger.info("Готово!")


if __name__ == "__main__":
    main()
