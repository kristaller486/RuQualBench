import argparse
import asyncio
import json
import logging
import os
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

import tiktoken
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from litellm import acompletion
import litellm
litellm.ssl_verify = False
# litellm._turn_on_debug()
from tqdm.asyncio import tqdm as atqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Отключаем verbose логи от litellm
for logger_name in ['litellm', 'LiteLLM']:
    litellm_logger = logging.getLogger(logger_name)
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

def load_dataset(dataset_name: str) -> List[List[Dict[str, str]]]:
    """Загружает датасет из json файла"""
    sizes = {'lite': 100, 'base': 250, 'large': 500}
    filename = f"{dataset_name}_bench_{sizes[dataset_name]}.json"
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def count_tokens(text: str) -> int:
    """Подсчитывает токены используя o200k_base encoding"""
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))

def extract_json_from_response(text: str) -> dict:
    """Извлекает JSON из ответа, который может быть обернут в markdown code block"""
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))
    
    return json.loads(text)

async def generate_answer(messages: List[Dict[str, str]], model_name: str, 
                         api_base: str, api_key: str, semaphore: asyncio.Semaphore,
                         max_retries: int, retry_delay: float, temperature: float = 1.0, 
                         max_tokens: int = 8192, extra_body: dict = None) -> str:
    """Генерирует ответ тестируемой модели асинхронно с ретраями"""
    async with semaphore:
        last_error = None
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "api_base": api_base,
                    "api_key": api_key,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                if extra_body:
                    kwargs["extra_body"] = extra_body
                    
                response = await acompletion(**kwargs)
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    raise last_error

async def judge_answer(dialog: List[Dict[str, str]], answer: str, 
                      judge_model: str, judge_api_base: str, judge_api_key: str,
                      semaphore: asyncio.Semaphore, max_retries: int, retry_delay: float,
                      extra_body: dict = None) -> Dict[str, int]:
    """Оценивает ответ с помощью модели-judge асинхронно с ретраями"""
    async with semaphore:
        env = Environment(loader=FileSystemLoader('prompts'))
        system_template = env.get_template('judge_system.jinja2')
        user_template = env.get_template('judge_user.jinja2')
        system_prompt = system_template.render()
        
        history_for_judge = dialog[:-1] if len(dialog) > 1 else None
        user_prompt = dialog[-1]["content"]
        
        user_content = user_template.render(
            history=history_for_judge,
            prompt=user_prompt,
            answer=answer
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        last_error = None
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": judge_model,
                    "messages": messages,
                    "api_base": judge_api_base,
                    "api_key": judge_api_key
                }
                if extra_body:
                    kwargs["extra_body"] = extra_body
                
                response = await acompletion(**kwargs)
                
                result = extract_json_from_response(response.choices[0].message.content)
                return {
                    "critical_mistakes": result.get("critical_mistakes", 0),
                    "mistakes": result.get("mistakes", 0),
                    "additional_mistakes": result.get("additional_mistakes", 0),
                    "explanation_critical_mistakes": result.get("explanation_critical_mistakes", []),
                    "explanation_mistakes": result.get("explanation_mistakes", []),
                    "explanation_additional_mistakes": result.get("explanation_additional_mistakes", [])
                }
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"Ошибка при оценке после {max_retries} попыток: {e}", exc_info=True)
                    return {
                        "critical_mistakes": 0,
                        "mistakes": 0,
                        "additional_mistakes": 0,
                        "explanation_critical_mistakes": [],
                        "explanation_mistakes": [],
                        "explanation_additional_mistakes": []
                    }

async def generate_dialog_answer(dialog: List[Dict[str, str]], dialog_id: int,
                                 test_model: str, test_api_base: str, test_api_key: str,
                                 test_semaphore: asyncio.Semaphore,
                                 max_retries: int, retry_delay: float, temperature: float = 1.0,
                                 max_tokens: int = 8192, extra_body: dict = None) -> Dict[str, Any]:
    """Генерирует ответ для диалога"""
    try:
        answer = await generate_answer(dialog, test_model, test_api_base, test_api_key, 
                                      test_semaphore, max_retries, retry_delay, temperature, max_tokens, extra_body)
        tokens = count_tokens(answer)
        
        return {
            "dialog_id": dialog_id,
            "dialog": dialog,
            "answer": answer,
            "tokens": tokens,
            "error": None
        }
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа для диалога {dialog_id}: {e}", exc_info=True)
        return {
            "dialog_id": dialog_id,
            "dialog": dialog,
            "answer": None,
            "tokens": 0,
            "error": str(e)
        }

async def judge_dialog_answer(answer_result: Dict[str, Any],
                             judge_model: str, judge_api_base: str, judge_api_key: str,
                             judge_semaphore: asyncio.Semaphore,
                             max_retries: int, retry_delay: float, extra_body: dict = None) -> Dict[str, Any]:
    """Оценивает готовый ответ диалога"""
    if answer_result["error"] is not None or answer_result["answer"] is None:
        return {
            **answer_result,
            "critical_mistakes": 0,
            "mistakes": 0,
            "additional_mistakes": 0,
            "explanation_critical_mistakes": [],
            "explanation_mistakes": [],
            "explanation_additional_mistakes": []
        }
    
    try:
        judge_result = await judge_answer(
            answer_result["dialog"], 
            answer_result["answer"], 
            judge_model, 
            judge_api_base, 
            judge_api_key, 
            judge_semaphore, 
            max_retries, 
            retry_delay,
            extra_body
        )
        
        return {
            **answer_result,
            "critical_mistakes": judge_result["critical_mistakes"],
            "mistakes": judge_result["mistakes"],
            "additional_mistakes": judge_result["additional_mistakes"],
            "explanation_critical_mistakes": judge_result["explanation_critical_mistakes"],
            "explanation_mistakes": judge_result["explanation_mistakes"],
            "explanation_additional_mistakes": judge_result["explanation_additional_mistakes"]
        }
    except Exception as e:
        logger.error(f"Ошибка при оценке диалога {answer_result['dialog_id']}: {e}", exc_info=True)
        return {
            **answer_result,
            "critical_mistakes": 0,
            "mistakes": 0,
            "additional_mistakes": 0,
            "explanation_critical_mistakes": [],
            "explanation_mistakes": [],
            "explanation_additional_mistakes": []
        }

async def judge_only_async(answer_results: List[Dict[str, Any]], 
                          judge_model_name: str = None,
                          run_number: int = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Оценивает готовые ответы судьей без генерации"""
    
    judge_model = judge_model_name or os.getenv("JUDGE_MODEL_NAME")
    judge_api_base = os.getenv("JUDGE_MODEL_BASE_URL")
    judge_api_key = os.getenv("JUDGE_MODEL_API_KEY")
    judge_max_workers = int(os.getenv("JUDGE_MODEL_MAX_WORKERS", "10"))
    
    judge_extra_body_str = os.getenv("JUDGE_MODEL_EXTRA_BODY")
    judge_extra_body = json.loads(judge_extra_body_str) if judge_extra_body_str else None
    
    max_retries = int(os.getenv("MAX_RETRIES", "3"))
    retry_delay = float(os.getenv("RETRY_DELAY", "1.0"))
    
    run_prefix = f"Прогон {run_number}: " if run_number is not None else ""
    
    logger.info(f"{run_prefix}Модель-оценщик: {judge_model} (max_workers={judge_max_workers})")
    if judge_extra_body:
        logger.info(f"{run_prefix}Judge extra body: {judge_extra_body}")
    
    judge_semaphore = asyncio.Semaphore(judge_max_workers)
    
    logger.info(f"{run_prefix}Оценка ответов судьей (без генерации)")
    judging_tasks = [
        judge_dialog_answer(
            answer_result,
            judge_model, judge_api_base, judge_api_key,
            judge_semaphore,
            max_retries, retry_delay, judge_extra_body
        )
        for answer_result in answer_results
    ]
    
    results = []
    for coro in atqdm.as_completed(judging_tasks, desc="Оценка ответов", total=len(judging_tasks)):
        result = await coro
        results.append(result)
    
    results.sort(key=lambda x: x["dialog_id"])
    
    valid_results = [r for r in results if r["error"] is None]
    total_critical_mistakes = sum(r["critical_mistakes"] for r in valid_results)
    total_mistakes = sum(r["mistakes"] for r in valid_results)
    total_additional_mistakes = sum(r["additional_mistakes"] for r in valid_results)
    total_all_mistakes = total_critical_mistakes + total_mistakes + total_additional_mistakes
    total_tokens = sum(r["tokens"] for r in valid_results)
    
    critical_mistakes_per_1000 = (total_critical_mistakes / total_tokens * 1000) if total_tokens > 0 else 0
    mistakes_per_1000 = (total_mistakes / total_tokens * 1000) if total_tokens > 0 else 0
    additional_mistakes_per_1000 = (total_additional_mistakes / total_tokens * 1000) if total_tokens > 0 else 0
    all_mistakes_per_1000 = (total_all_mistakes / total_tokens * 1000) if total_tokens > 0 else 0
    
    summary = {
        "total_critical_mistakes": total_critical_mistakes,
        "total_mistakes": total_mistakes,
        "total_additional_mistakes": total_additional_mistakes,
        "total_all_mistakes": total_all_mistakes,
        "total_tokens": total_tokens,
        "critical_mistakes_per_1000_tokens": round(critical_mistakes_per_1000, 2),
        "mistakes_per_1000_tokens": round(mistakes_per_1000, 2),
        "additional_mistakes_per_1000_tokens": round(additional_mistakes_per_1000, 2),
        "all_mistakes_per_1000_tokens": round(all_mistakes_per_1000, 2),
        "total_dialogs": len(answer_results),
        "successful_dialogs": len(valid_results),
        "failed_dialogs": len(answer_results) - len(valid_results)
    }
    
    return results, summary

async def run_benchmark_async(dataset_name: str, model_name: str = None, judge_model_name: str = None, 
                             extra_body: dict = None, run_number: int = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Запускает бенчмарк асинхронно и возвращает результаты"""
    
    test_model = model_name or os.getenv("TEST_MODEL_NAME")
    test_api_base = os.getenv("TEST_MODEL_BASE_URL")
    test_api_key = os.getenv("TEST_MODEL_API_KEY")
    test_max_workers = int(os.getenv("TEST_MODEL_MAX_WORKERS", "10"))
    test_temperature = float(os.getenv("TEST_MODEL_TEMPERATURE", "1.0"))
    test_max_tokens = int(os.getenv("TEST_MODEL_MAX_TOKENS", "8192"))
    
    judge_model = judge_model_name or os.getenv("JUDGE_MODEL_NAME")
    judge_api_base = os.getenv("JUDGE_MODEL_BASE_URL")
    judge_api_key = os.getenv("JUDGE_MODEL_API_KEY")
    judge_max_workers = int(os.getenv("JUDGE_MODEL_MAX_WORKERS", "10"))
    
    judge_extra_body_str = os.getenv("JUDGE_MODEL_EXTRA_BODY")
    judge_extra_body = json.loads(judge_extra_body_str) if judge_extra_body_str else None
    
    max_retries = int(os.getenv("MAX_RETRIES", "3"))
    retry_delay = float(os.getenv("RETRY_DELAY", "1.0"))
    
    run_prefix = f"Прогон {run_number}: " if run_number is not None else ""
    
    logger.info(f"{run_prefix}Тестируемая модель: {test_model} (max_workers={test_max_workers}, temperature={test_temperature})")
    if extra_body:
        logger.info(f"{run_prefix}Test extra body: {extra_body}")
    logger.info(f"{run_prefix}Модель-оценщик: {judge_model} (max_workers={judge_max_workers})")
    if judge_extra_body:
        logger.info(f"{run_prefix}Judge extra body: {judge_extra_body}")
    logger.info(f"{run_prefix}Настройки ретраев: max_retries={max_retries}, retry_delay={retry_delay}")
    logger.info(f"{run_prefix}Датасет: {dataset_name}")
    
    dataset = load_dataset(dataset_name)
    
    test_semaphore = asyncio.Semaphore(test_max_workers)
    judge_semaphore = asyncio.Semaphore(judge_max_workers)
    
    logger.info(f"{run_prefix}Этап 1: Генерация ответов от тестируемой модели")
    generation_tasks = [
        generate_dialog_answer(
            dialog, i,
            test_model, test_api_base, test_api_key,
            test_semaphore,
            max_retries, retry_delay, test_temperature, test_max_tokens, extra_body
        )
        for i, dialog in enumerate(dataset)
    ]
    
    answer_results = []
    for coro in atqdm.as_completed(generation_tasks, desc="Генерация ответов", total=len(generation_tasks)):
        result = await coro
        answer_results.append(result)
    
    answer_results.sort(key=lambda x: x["dialog_id"])
    
    logger.info(f"{run_prefix}Этап 2: Оценка ответов судьей")
    judging_tasks = [
        judge_dialog_answer(
            answer_result,
            judge_model, judge_api_base, judge_api_key,
            judge_semaphore,
            max_retries, retry_delay, judge_extra_body
        )
        for answer_result in answer_results
    ]
    
    results = []
    for coro in atqdm.as_completed(judging_tasks, desc="Оценка ответов", total=len(judging_tasks)):
        result = await coro
        results.append(result)
    
    results.sort(key=lambda x: x["dialog_id"])
    
    valid_results = [r for r in results if r["error"] is None]
    total_critical_mistakes = sum(r["critical_mistakes"] for r in valid_results)
    total_mistakes = sum(r["mistakes"] for r in valid_results)
    total_additional_mistakes = sum(r["additional_mistakes"] for r in valid_results)
    total_all_mistakes = total_critical_mistakes + total_mistakes + total_additional_mistakes
    total_tokens = sum(r["tokens"] for r in valid_results)
    
    critical_mistakes_per_1000 = (total_critical_mistakes / total_tokens * 1000) if total_tokens > 0 else 0
    mistakes_per_1000 = (total_mistakes / total_tokens * 1000) if total_tokens > 0 else 0
    additional_mistakes_per_1000 = (total_additional_mistakes / total_tokens * 1000) if total_tokens > 0 else 0
    all_mistakes_per_1000 = (total_all_mistakes / total_tokens * 1000) if total_tokens > 0 else 0
    
    summary = {
        "total_critical_mistakes": total_critical_mistakes,
        "total_mistakes": total_mistakes,
        "total_additional_mistakes": total_additional_mistakes,
        "total_all_mistakes": total_all_mistakes,
        "total_tokens": total_tokens,
        "critical_mistakes_per_1000_tokens": round(critical_mistakes_per_1000, 2),
        "mistakes_per_1000_tokens": round(mistakes_per_1000, 2),
        "additional_mistakes_per_1000_tokens": round(additional_mistakes_per_1000, 2),
        "all_mistakes_per_1000_tokens": round(all_mistakes_per_1000, 2),
        "total_dialogs": len(dataset),
        "successful_dialogs": len(valid_results),
        "failed_dialogs": len(dataset) - len(valid_results)
    }
    
    config = {
        "dataset": dataset_name,
        "model": test_model,
        "judge_model": judge_model,
        "test_max_workers": test_max_workers,
        "judge_max_workers": judge_max_workers
    }
    
    return results, summary, config

def find_existing_runs(timestamp: str, dataset: str) -> tuple:
    """Находит существующие прогоны с указанным timestamp"""
    logs_dir = Path("logs")
    pattern = f"benchmark_{timestamp}_run_*_{dataset}.json"
    
    existing_files = list(logs_dir.glob(pattern))
    
    if not existing_files:
        raise ValueError(f"Не найдено логов с timestamp {timestamp} и dataset {dataset}")
    
    # Извлекаем номера run
    run_numbers = []
    for file in existing_files:
        match = re.search(r'_run_(\d+)_', file.name)
        if match:
            run_numbers.append(int(match.group(1)))
    
    if not run_numbers:
        raise ValueError(f"Не удалось извлечь номера run из файлов")
    
    max_run = max(run_numbers)
    
    # Читаем config и results из первого прогона
    first_run_file = logs_dir / f"benchmark_{timestamp}_run_1_{dataset}.json"
    with open(first_run_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        config = data.get("config", {})
        results = data.get("results", [])
    
    logger.info(f"Найдено {len(run_numbers)} существующих прогонов (максимальный: run_{max_run})")
    
    return max_run, config, results

async def run_multiple_benchmarks_async(dataset_name: str, num_runs: int, model_name: str = None,
                                       judge_model_name: str = None, extra_body: dict = None, 
                                       verbose_name: str = None, continue_timestamp: str = None,
                                       start_run_number: int = 0, no_regenerate: bool = False,
                                       existing_answer_results: List[Dict[str, Any]] = None):
    """Запускает бенчмарк несколько раз и вычисляет статистику"""
    
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    if continue_timestamp:
        base_timestamp = continue_timestamp
        start_num = start_run_number + 1
        end_num = start_run_number + num_runs
    else:
        base_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        start_num = 1
        end_num = num_runs
    
    all_summaries = []
    shared_answer_results = None
    
    logger.info("="*70)
    if continue_timestamp:
        logger.info(f"ПРОДОЛЖЕНИЕ ПРОГОНОВ (добавление {num_runs} прогонов к существующим)")
        logger.info(f"Timestamp: {continue_timestamp}")
    else:
        logger.info(f"ЗАПУСК МНОЖЕСТВЕННЫХ ПРОГОНОВ (N={num_runs})")
    
    if no_regenerate:
        logger.info(f"РЕЖИМ: --no-regenerate (генерация ответов один раз, оценка судьей {num_runs} раз)")
    logger.info("="*70)
    
    # Если no_regenerate и есть готовые ответы (из --continue), используем их
    if no_regenerate and existing_answer_results:
        logger.info("Используются ответы из run_1 существующей серии")
        shared_answer_results = existing_answer_results
    # Если no_regenerate но нет готовых ответов, генерируем один раз перед циклом
    elif no_regenerate and not existing_answer_results:
        logger.info("")
        logger.info("="*70)
        logger.info("ГЕНЕРАЦИЯ ОТВЕТОВ (один раз для всех прогонов)")
        logger.info("="*70)
        
        results, summary, config = await run_benchmark_async(
            dataset_name, model_name, judge_model_name, extra_body, run_number=None
        )
        
        # Извлекаем только ответы без оценок судьи
        shared_answer_results = [
            {
                "dialog_id": r["dialog_id"],
                "dialog": r["dialog"],
                "answer": r["answer"],
                "tokens": r["tokens"],
                "error": r["error"]
            }
            for r in results
        ]
        
        logger.info(f"Сгенерировано {len(shared_answer_results)} ответов")
    
    for run_num in range(start_num, end_num + 1):
        logger.info("")
        logger.info(f"{'='*70}")
        logger.info(f"ПРОГОН {run_num}/{end_num}")
        logger.info(f"{'='*70}")
        
        # Если no_regenerate, используем готовые ответы и только оцениваем судьей
        if no_regenerate and shared_answer_results:
            results, summary = await judge_only_async(
                shared_answer_results, judge_model_name, run_number=run_num
            )
            
            # Получаем config из первой генерации или из существующих логов
            if existing_answer_results:
                # При --continue берем config из существующих логов
                first_run_file = logs_dir / f"benchmark_{base_timestamp}_run_1_{dataset_name}.json"
                with open(first_run_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    config = data.get("config", {})
                    # Убираем поля специфичные для конкретного прогона
                    config = {k: v for k, v in config.items() if k not in ['timestamp', 'run_number', 'total_runs']}
            else:
                # При первом запуске создаем config
                test_model = model_name or os.getenv("TEST_MODEL_NAME")
                judge_model = judge_model_name or os.getenv("JUDGE_MODEL_NAME")
                test_max_workers = int(os.getenv("TEST_MODEL_MAX_WORKERS", "10"))
                judge_max_workers = int(os.getenv("JUDGE_MODEL_MAX_WORKERS", "10"))
                
                config = {
                    "dataset": dataset_name,
                    "model": test_model,
                    "judge_model": judge_model,
                    "test_max_workers": test_max_workers,
                    "judge_max_workers": judge_max_workers,
                    "no_regenerate": True
                }
        else:
            # Обычный режим: генерация + оценка
            results, summary, config = await run_benchmark_async(
                dataset_name, model_name, judge_model_name, extra_body, run_number=run_num
            )
        
        all_summaries.append(summary)
        
        log_filename = logs_dir / f"benchmark_{base_timestamp}_run_{run_num}_{dataset_name}.json"
        
        log_data = {
            "config": {
                **config,
                "timestamp": base_timestamp,
                "run_number": run_num,
                "total_runs": end_num
            },
            "results": results,
            "summary": summary
        }
        
        if verbose_name:
            log_data["config"]["verbose_name"] = verbose_name
        
        with open(log_filename, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Прогон {run_num}: Обработано {summary['successful_dialogs']}/{summary['total_dialogs']} диалогов")
        logger.info(f"Прогон {run_num}: Результаты сохранены в {log_filename.as_posix()}")
    
    if num_runs > 1:
        logger.info("")
        logger.info("="*70)
        logger.info("АГРЕГИРОВАННЫЕ РЕЗУЛЬТАТЫ")
        logger.info("="*70)
        
        metrics_to_aggregate = [
            'critical_mistakes_per_1000_tokens',
            'mistakes_per_1000_tokens',
            'additional_mistakes_per_1000_tokens',
            'all_mistakes_per_1000_tokens'
        ]
        
        for metric in metrics_to_aggregate:
            values = [s[metric] for s in all_summaries]
            mean_val = statistics.mean(values)
            
            if num_runs >= 2:
                stdev_val = statistics.stdev(values) if num_runs > 1 else 0
                stderr_val = stdev_val / (num_runs ** 0.5)
                
                metric_name = metric.replace('_', ' ').replace(' per 1000 tokens', '')
                logger.info(f"{metric_name.capitalize()} на 1000 токенов:")
                logger.info(f"  Среднее: {mean_val:.2f} ± {stderr_val:.2f} (SE)")
                logger.info(f"  Стандартное отклонение: {stdev_val:.2f}")
                logger.info(f"  Диапазон: [{min(values):.2f}, {max(values):.2f}]")
            else:
                metric_name = metric.replace('_', ' ').replace(' per 1000 tokens', '')
                logger.info(f"{metric_name.capitalize()} на 1000 токенов: {mean_val:.2f}")
        
        avg_tokens = statistics.mean([s['total_tokens'] for s in all_summaries])
        logger.info(f"\nСреднее количество токенов: {avg_tokens:.0f}")
        logger.info("="*70)

def run_benchmark(dataset_name: str, num_runs: int = 1, model_name: str = None, 
                 judge_model_name: str = None, extra_body: dict = None, verbose_name: str = None,
                 continue_timestamp: str = None, start_run_number: int = 0, no_regenerate: bool = False,
                 existing_answer_results: List[Dict[str, Any]] = None):
    """Обертка для запуска бенчмарка (одного или нескольких прогонов)"""
    asyncio.run(run_multiple_benchmarks_async(dataset_name, num_runs, model_name, judge_model_name, 
                                              extra_body, verbose_name, continue_timestamp, start_run_number,
                                              no_regenerate, existing_answer_results))

def main():
    parser = argparse.ArgumentParser(description="RuQualBench - бенчмарк качества русского языка")
    parser.add_argument(
        "--dataset",
        type=str,
        default="lite",
        choices=["lite", "base", "large"],
        help="Выбор датасета (по умолчанию: lite)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Переопределить тестируемую модель из .env"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        help="Переопределить модель-оценщик из .env"
    )
    parser.add_argument(
        "--extra-body",
        type=str,
        help='JSON объект для extra_body параметра тестируемой модели (например: \'{"temperature": 0.7}\')'
    )
    parser.add_argument(
        "-n", "--num-runs",
        type=int,
        default=1,
        help="Количество прогонов бенчмарка для вычисления средних значений и погрешности (по умолчанию: 1)"
    )
    parser.add_argument(
        "-v", "--verbose-name",
        type=str,
        help="Красивое имя модели для отображения в лидерборде (опционально)"
    )
    parser.add_argument(
        "--continue",
        type=str,
        dest="continue_timestamp",
        help="Продолжить существующую серию прогонов (указать timestamp, например: 2025-10-17_15-17-05)"
    )
    parser.add_argument(
        "--no-regenerate",
        action="store_true",
        help="Генерировать ответы от модели только один раз, оценивать судьей N раз (работает с -n)"
    )
    
    args = parser.parse_args()
    
    extra_body = None
    if args.extra_body:
        extra_body = json.loads(args.extra_body)
    
    # Обработка режима продолжения
    if args.continue_timestamp:
        max_run, existing_config, existing_results = find_existing_runs(args.continue_timestamp, args.dataset)
        
        # Берем параметры из существующих логов
        model = existing_config.get("model")
        judge_model = existing_config.get("judge_model")
        verbose_name = existing_config.get("verbose_name")
        existing_no_regenerate = existing_config.get("no_regenerate", False)
        
        logger.info(f"Продолжение серии с моделью: {model}")
        if verbose_name:
            logger.info(f"Verbose name: {verbose_name}")
        
        # Определяем режим no_regenerate
        if args.no_regenerate and not existing_no_regenerate:
            logger.warning("ВНИМАНИЕ: --no-regenerate указан, но исходная серия запускалась без этого флага")
            logger.warning("Будет использован режим --no-regenerate с ответами из run_1 исходной серии")
            use_no_regenerate = True
        elif existing_no_regenerate:
            logger.info("Исходная серия использовала --no-regenerate, продолжаем в том же режиме")
            use_no_regenerate = True
        else:
            use_no_regenerate = False
        
        # Извлекаем только ответы без оценок судьи из существующих результатов
        answer_results = None
        if use_no_regenerate and existing_results:
            answer_results = [
                {
                    "dialog_id": r["dialog_id"],
                    "dialog": r["dialog"],
                    "answer": r["answer"],
                    "tokens": r["tokens"],
                    "error": r["error"]
                }
                for r in existing_results
            ]
        
        run_benchmark(
            args.dataset, 
            args.num_runs, 
            model, 
            judge_model, 
            extra_body, 
            verbose_name,
            args.continue_timestamp,
            max_run,
            use_no_regenerate,
            answer_results
        )
    else:
        run_benchmark(
            args.dataset, 
            args.num_runs, 
            args.model, 
            args.judge_model, 
            extra_body, 
            args.verbose_name,
            no_regenerate=args.no_regenerate
        )

if __name__ == "__main__":
    main()
