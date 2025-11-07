import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

from datasets import Dataset
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from litellm import acompletion
import litellm
from tqdm.asyncio import tqdm as atqdm

litellm.ssl_verify = False

load_dotenv()

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
    
    # Ранний выход после нахождения достаточного количества повторов
    count = 0
    for match in re.finditer(pattern, text, re.DOTALL):
        count += 1
        if count > 5:
            return True
    
    return False


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


def hash_correction_input(answer: str, mistakes: List[str]) -> str:
    """Генерирует хеш для пары (ответ, список ошибок)"""
    content = answer + "|" + "|".join(sorted(mistakes))
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def load_log_cache(log_file: str, cache_dir: Path) -> dict:
    """Загружает кеш для конкретного лог-файла"""
    cache_path = cache_dir / log_file
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_log_cache(log_file: str, cache_dir: Path, cache: dict):
    """Сохраняет кеш для конкретного лог-файла"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / log_file
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


async def correct_mistakes(answer: str, prompt: Any, mistakes: List[str], 
                          model_name: str, api_base: str, api_key: str,
                          semaphore: asyncio.Semaphore, max_retries: int, 
                          retry_delay: float, temperature: float = 0.3,
                          max_tokens: int = 8192) -> str:
    """Исправляет ошибки в ответе с помощью модели"""
    async with semaphore:
        env = Environment(loader=FileSystemLoader('prompts'))
        system_template = env.get_template('correct_mistakes_system.jinja2')
        user_template = env.get_template('correct_mistakes_user.jinja2')
        
        system_prompt = system_template.render()
        user_prompt = user_template.render(
            answer=answer,
            prompt=prompt,
            mistakes=mistakes
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await acompletion(
                    model=model_name,
                    messages=messages,
                    api_base=api_base,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    raise last_error


async def correct_dataset_mistakes(dataset_dict: Dict[str, List[Any]], 
                                   log_data_list: List[Dict[str, Any]],
                                   cache_dir: Path) -> Dict[str, List[Any]]:
    """Добавляет колонку с исправленными текстами в датасет"""
    
    correct_model = os.getenv("CORRECT_MODEL_NAME")
    correct_api_base = os.getenv("CORRECT_MODEL_BASE_URL")
    correct_api_key = os.getenv("CORRECT_MODEL_API_KEY")
    correct_max_workers = int(os.getenv("CORRECT_MODEL_MAX_WORKERS", "5"))
    correct_temperature = float(os.getenv("CORRECT_MODEL_TEMPERATURE", "0.3"))
    correct_max_tokens = int(os.getenv("CORRECT_MODEL_MAX_TOKENS", "8192"))
    
    max_retries = int(os.getenv("MAX_RETRIES", "3"))
    retry_delay = float(os.getenv("RETRY_DELAY", "1.0"))
    
    logger.info(f"Модель исправления: {correct_model} (max_workers={correct_max_workers}, temperature={correct_temperature})")
    
    semaphore = asyncio.Semaphore(correct_max_workers)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем маппинг: (timestamp, run_number) -> log_file для поиска кеша
    log_file_map = {}
    for log_item in log_data_list:
        log_file = log_item["file"]
        config = log_item["data"].get("config", {})
        timestamp = config.get("timestamp", "unknown")
        run_number = config.get("run_number", 0)
        log_file_map[(timestamp, run_number)] = log_file
    
    # Группируем записи по лог-файлам для батчевой обработки
    records_by_log = defaultdict(list)
    for i in range(len(dataset_dict["dialog_id"])):
        timestamp = dataset_dict["timestamp"][i]
        run_number = dataset_dict["run_number"][i]
        log_file = log_file_map.get((timestamp, run_number), "unknown.json")
        records_by_log[log_file].append(i)
    
    corrected_texts = [None] * len(dataset_dict["dialog_id"])
    
    total_to_generate = 0
    total_from_cache = 0
    
    # Обрабатываем каждый лог-файл
    for log_file, record_indices in records_by_log.items():
        logger.info(f"Обработка {log_file}: {len(record_indices)} записей")
        
        # Загружаем кеш для этого лог-файла
        cache = load_log_cache(log_file, cache_dir)
        cache_modified = False
        
        tasks = []
        
        for idx in record_indices:
            test_answer = dataset_dict["test_answer"][idx]
            dialog = dataset_dict["dialog"][idx]
            
            # Edge cases
            if not test_answer:
                corrected_texts[idx] = None
                continue
            
            # Собираем все ошибки
            all_mistakes = (
                dataset_dict["explanation_critical_mistakes"][idx] +
                dataset_dict["explanation_mistakes"][idx] +
                dataset_dict["explanation_additional_mistakes"][idx]
            )
            
            # Если ошибок нет, просто копируем
            if not all_mistakes:
                corrected_texts[idx] = test_answer
                continue
            
            # Вычисляем хеш
            cache_hash = hash_correction_input(test_answer, all_mistakes)
            
            # Проверяем кеш
            if cache_hash in cache:
                corrected_texts[idx] = cache[cache_hash]
                total_from_cache += 1
            else:
                # Создаем задачу для генерации
                tasks.append((idx, test_answer, dialog, all_mistakes, cache_hash))
                total_to_generate += 1
        
        # Генерируем исправления для новых записей
        if tasks:
            logger.info(f"  Генерация исправлений: {len(tasks)} (из кеша: {len(record_indices) - len(tasks)})")
            
            async_tasks = [
                correct_mistakes(
                    answer=task[1],
                    prompt=task[2],
                    mistakes=task[3],
                    model_name=correct_model,
                    api_base=correct_api_base,
                    api_key=correct_api_key,
                    semaphore=semaphore,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    temperature=correct_temperature,
                    max_tokens=correct_max_tokens
                )
                for task in tasks
            ]
            
            results = []
            for coro in atqdm.as_completed(async_tasks, desc=f"Исправление ({log_file})", total=len(async_tasks)):
                try:
                    result = await coro
                    results.append(result)
                except Exception as e:
                    logger.error(f"Ошибка при исправлении: {e}", exc_info=True)
                    results.append(None)
            
            # Сохраняем результаты
            for (idx, _, _, _, cache_hash), corrected_text in zip(tasks, results):
                corrected_texts[idx] = corrected_text
                if corrected_text is not None:
                    cache[cache_hash] = corrected_text
                    cache_modified = True
        
        # Сохраняем обновленный кеш
        if cache_modified:
            save_log_cache(log_file, cache_dir, cache)
            logger.info(f"  Кеш обновлен: {len(cache)} записей")
    
    logger.info(f"Всего исправлений: {total_to_generate} сгенерировано, {total_from_cache} из кеша")
    
    # Добавляем колонку в датасет
    dataset_dict["corrected_text"] = corrected_texts
    
    return dataset_dict


def prepare_kto_dataset(dataset_dict: Dict[str, List[Any]], 
                        balanced: bool = False,
                        difficulty: Optional[str] = None) -> Dict[str, List[Any]]:
    """Преобразует датасет в формат KTO (prompt/completion/label)
    
    Args:
        dataset_dict: Исходный датасет
        balanced: Балансировать ли выборку (50/50 true/false)
        difficulty: Сложность false примеров для balanced ('easy'/'medium'/'hard')
    """
    
    logger.info("")
    logger.info("="*70)
    logger.info("СОЗДАНИЕ KTO ДАТАСЕТА")
    logger.info("="*70)
    
    kto_dict = {
        "prompt": [],
        "completion": [],
        "label": []
    }
    
    # Фильтруем: исключаем циклы генерации
    valid_indices = []
    for i in range(len(dataset_dict["dialog_id"])):
        if not dataset_dict["is_generation_cycle"][i]:
            valid_indices.append(i)
    
    logger.info(f"Исключено примеров с циклами генерации: {len(dataset_dict['dialog_id']) - len(valid_indices)}")
    logger.info(f"Осталось примеров: {len(valid_indices)}")
    
    # Разделяем на true и false примеры
    true_indices = []
    false_indices = []
    
    for i in valid_indices:
        # label=True: нет ошибок вообще И нет ошибок выполнения
        total_mistakes = (
            dataset_dict["critical_mistakes"][i] +
            dataset_dict["mistakes"][i] +
            dataset_dict["additional_mistakes"][i]
        )
        
        if total_mistakes == 0 and not dataset_dict["has_error"][i]:
            true_indices.append(i)
        else:
            false_indices.append(i)
    
    logger.info(f"Примеров с label=True: {len(true_indices)}")
    logger.info(f"Примеров с label=False: {len(false_indices)}")
    
    # Применяем балансировку
    selected_true = true_indices
    selected_false = false_indices
    
    if balanced:
        logger.info("")
        logger.info("Применение балансировки датасета...")
        
        num_true = len(true_indices)
        num_false = len(false_indices)
        
        if num_true > num_false:
            logger.warning(f"True примеров ({num_true}) больше чем false ({num_false})")
            logger.warning("Балансировка будет выполнена по количеству false примеров")
            target_count = num_false
            selected_true = random.sample(true_indices, target_count)
            selected_false = false_indices
        else:
            target_count = num_true
            selected_true = true_indices
            
            # Применяем фильтрацию по сложности для false примеров
            if difficulty:
                logger.info(f"Фильтрация false примеров по сложности: {difficulty}")
                
                # Вычисляем total_mistakes для всех false примеров
                false_with_mistakes = []
                for idx in false_indices:
                    total_mistakes = (
                        dataset_dict["critical_mistakes"][idx] +
                        dataset_dict["mistakes"][idx] +
                        dataset_dict["additional_mistakes"][idx]
                    )
                    false_with_mistakes.append((idx, total_mistakes))
                
                if difficulty == "easy":
                    # Большое количество ошибок - сортируем по убыванию
                    false_with_mistakes.sort(key=lambda x: x[1], reverse=True)
                    selected_false = [idx for idx, _ in false_with_mistakes[:target_count]]
                    mistakes_range = [m for _, m in false_with_mistakes[:target_count]]
                    logger.info(f"  Easy: выбраны примеры с {min(mistakes_range)}-{max(mistakes_range)} ошибками")
                    
                elif difficulty == "hard":
                    # Небольшое количество ошибок - сортируем по возрастанию
                    false_with_mistakes.sort(key=lambda x: x[1])
                    selected_false = [idx for idx, _ in false_with_mistakes[:target_count]]
                    mistakes_range = [m for _, m in false_with_mistakes[:target_count]]
                    logger.info(f"  Hard: выбраны примеры с {min(mistakes_range)}-{max(mistakes_range)} ошибками")
                    
                elif difficulty == "medium":
                    # Среднее количество ошибок - берем вокруг медианы
                    mistakes_values = [m for _, m in false_with_mistakes]
                    median_mistakes = statistics.median(mistakes_values)
                    
                    # Вычисляем диапазон: ±25% от медианы
                    lower_bound = median_mistakes * 0.75
                    upper_bound = median_mistakes * 1.25
                    
                    # Фильтруем примеры в диапазоне
                    medium_candidates = [
                        (idx, m) for idx, m in false_with_mistakes 
                        if lower_bound <= m <= upper_bound
                    ]
                    
                    if len(medium_candidates) >= target_count:
                        selected_false = [idx for idx, _ in random.sample(medium_candidates, target_count)]
                        mistakes_range = [m for _, m in medium_candidates[:target_count]]
                        logger.info(f"  Medium: выбраны примеры с {min(mistakes_range)}-{max(mistakes_range)} ошибками")
                        logger.info(f"  Медиана ошибок: {median_mistakes:.1f}, диапазон: [{lower_bound:.1f}, {upper_bound:.1f}]")
                    else:
                        logger.warning(f"  Недостаточно примеров в среднем диапазоне ({len(medium_candidates)} < {target_count})")
                        logger.warning(f"  Используется случайная выборка из всех false примеров")
                        selected_false = random.sample(false_indices, target_count)
            else:
                # Случайная выборка без учета сложности
                selected_false = random.sample(false_indices, target_count)
        
        logger.info(f"После балансировки: {len(selected_true)} true, {len(selected_false)} false")
    
    # Формируем KTO датасет
    all_selected = [(idx, True) for idx in selected_true] + [(idx, False) for idx in selected_false]
    random.shuffle(all_selected)
    
    for idx, label in all_selected:
        kto_dict["prompt"].append(dataset_dict["dialog"][idx])
        # completion как список с одним сообщением в формате chat
        kto_dict["completion"].append([{
            "role": "assistant",
            "content": dataset_dict["test_answer"][idx]
        }])
        kto_dict["label"].append(label)
    
    logger.info(f"Итоговый KTO датасет: {len(kto_dict['prompt'])} примеров")
    logger.info(f"  True: {sum(kto_dict['label'])}")
    logger.info(f"  False: {len(kto_dict['label']) - sum(kto_dict['label'])}")
    
    return kto_dict


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
        # Удаляем существующую директорию если есть
        if output_path.exists():
            import shutil
            shutil.rmtree(output_path)
            logger.info(f"Удалена существующая директория {output_path}")
        dataset.save_to_disk(str(output_path))
        logger.info(f"Датасет сохранен в {output_path} (формат: arrow)")
    elif output_path.suffix == '.jsonl':
        dataset.to_json(str(output_path), force_ascii=False)
        logger.info(f"Датасет сохранен в {output_path} (формат: jsonl)")
    else:
        # По умолчанию сохраняем как arrow (директория)
        # Удаляем существующую директорию если есть
        if output_path.exists():
            import shutil
            shutil.rmtree(output_path)
            logger.info(f"Удалена существующая директория {output_path}")
        dataset.save_to_disk(str(output_path))
        logger.info(f"Датасет сохранен в {output_path} (формат: arrow)")


async def main_async():
    parser = argparse.ArgumentParser(
        description="Извлекает датасет из файлов логов бенчмарка"
    )
    
    # Общие параметры
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Директория с файлами логов (по умолчанию: logs)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Путь для сохранения датасета"
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
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Режим работы")
    
    # Команда: default (без subcommand)
    parser.add_argument(
        "--correct-mistakes",
        action="store_true",
        help="Добавить колонку с исправленными текстами (используя CORRECT_MODEL_* из .env)"
    )
    
    # Команда: kto
    kto_parser = subparsers.add_parser(
        "kto",
        help="Создать датасет в формате KTO (prompt/completion/label)"
    )
    kto_parser.add_argument(
        "--balanced",
        action="store_true",
        help="Балансировать датасет (50/50 true/false примеров)"
    )
    kto_parser.add_argument(
        "--easy",
        action="store_true",
        help="При --balanced выбирать false примеры с большим количеством ошибок"
    )
    kto_parser.add_argument(
        "--medium",
        action="store_true",
        help="При --balanced выбирать false примеры со средним количеством ошибок"
    )
    kto_parser.add_argument(
        "--hard",
        action="store_true",
        help="При --balanced выбирать false примеры с малым количеством ошибок"
    )
    
    args = parser.parse_args()
    
    logs_dir = Path(args.logs_dir)
    
    # Определяем режим работы и output path
    is_kto_mode = args.command == "kto"
    
    if args.output:
        output_path = Path(args.output)
    else:
        # Выбираем дефолтное имя в зависимости от режима
        if is_kto_mode:
            output_path = Path("kto_dataset.parquet")
        else:
            output_path = Path("extracted_dataset.parquet")
    
    if not logs_dir.exists():
        logger.error(f"Директория {logs_dir} не существует")
        return
    
    logger.info("Начало извлечения датасета")
    logger.info(f"Директория логов: {logs_dir}")
    logger.info(f"Выходной файл: {output_path}")
    
    if is_kto_mode:
        logger.info("Режим: KTO датасет")
    
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
    
    # Обработка в зависимости от режима
    if is_kto_mode:
        # Проверка флагов сложности
        difficulty_flags = sum([args.easy, args.medium, args.hard])
        if difficulty_flags > 1:
            logger.error("Ошибка: можно указать только один флаг сложности (--easy, --medium, или --hard)")
            return
        
        if difficulty_flags > 0 and not args.balanced:
            logger.error("Ошибка: флаги сложности (--easy/--medium/--hard) работают только с --balanced")
            return
        
        # Определяем сложность
        difficulty = None
        if args.easy:
            difficulty = "easy"
        elif args.medium:
            difficulty = "medium"
        elif args.hard:
            difficulty = "hard"
        
        # Создаем KTO датасет
        kto_dict = prepare_kto_dataset(dataset_dict, args.balanced, difficulty)
        save_dataset(kto_dict, output_path)
    else:
        # Исправление ошибок (если включен флаг)
        if args.correct_mistakes:
            logger.info("")
            logger.info("="*70)
            logger.info("ИСПРАВЛЕНИЕ ОШИБОК")
            logger.info("="*70)
            
            cache_dir = Path("cache/correct_mistakes")
            dataset_dict = await correct_dataset_mistakes(dataset_dict, log_data_list, cache_dir)
        
        # Сохранение датасета
        save_dataset(dataset_dict, output_path)
    
    logger.info("Готово!")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
