import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path

import litellm
from dotenv import load_dotenv

from benchmark.v1 import BenchmarkV1
from benchmark.v2 import BenchmarkV2

# Настройка окружения
litellm.ssl_verify = False
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

def main():
    parser = argparse.ArgumentParser(description="RuQualBench - бенчмарк качества русского языка")
    subparsers = parser.add_subparsers(dest="command", help="Версия бенчмарка")

    # Парсер для v1 (по умолчанию)
    parser_v1 = subparsers.add_parser("v1", help="Запуск версии v1")
    
    # Парсер для v2
    parser_v2 = subparsers.add_parser("v2", help="Запуск версии v2")

    # Общие аргументы
    for p in [parser_v1, parser_v2]:
        p.add_argument(
            "--dataset",
            type=str,
            default="lite",
            choices=["debug", "lite", "base", "large"],
            help="Выбор датасета (по умолчанию: lite)"
        )
        p.add_argument(
            "--model",
            type=str,
            help="Переопределить тестируемую модель из .env"
        )
        p.add_argument(
            "--judge-model",
            type=str,
            help="Переопределить модель-оценщик из .env"
        )
        p.add_argument(
            "--extra-body",
            type=str,
            help='JSON объект для extra_body параметра тестируемой модели (например: \'{"temperature": 0.7}\')'
        )
        p.add_argument(
            "-n", "--num-runs",
            type=int,
            default=1,
            help="Количество прогонов бенчмарка для вычисления средних значений и погрешности (по умолчанию: 1)"
        )
        p.add_argument(
            "-v", "--verbose-name",
            type=str,
            help="Красивое имя модели для отображения в лидерборде (опционально)"
        )

    # Аргументы только для v1
    parser_v1.add_argument(
        "--continue",
        type=str,
        dest="continue_timestamp",
        help="Продолжить существующую серию прогонов (указать timestamp, например: 2025-10-17_15-17-05)"
    )
    parser_v1.add_argument(
        "--no-regenerate",
        action="store_true",
        help="Генерировать ответы от модели только один раз, оценивать судьей N раз (работает с -n)"
    )

    # Аргументы только для v2
    parser_v2.add_argument(
        "--debug-logs",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Уровень отладки логов: 0 - стандартный, 1 - добавить сырой ответ судьи, 2 - добавить сырой ответ и промпт"
    )

    # Если аргументы не переданы или первый аргумент не v1/v2, считаем что это v1
    import sys
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] not in ["v1", "v2"]):
        # Вставляем 'v1' первым аргументом
        sys.argv.insert(1, "v1")

    args = parser.parse_args()
    
    extra_body = None
    if args.extra_body:
        extra_body = json.loads(args.extra_body)
    
    if args.command == "v2":
        benchmark = BenchmarkV2(
            dataset_name=args.dataset,
            model_name=args.model,
            judge_model_name=args.judge_model,
            extra_body=extra_body,
            verbose_name=args.verbose_name,
            debug_logs=args.debug_logs
        )
        asyncio.run(benchmark.run_multiple_benchmarks(args.num_runs))
    else:
        # Логика v1
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
            
            benchmark = BenchmarkV1(
                dataset_name=args.dataset,
                model_name=model,
                judge_model_name=judge_model,
                extra_body=extra_body,
                verbose_name=verbose_name
            )
            
            asyncio.run(benchmark.run_multiple_benchmarks(
                num_runs=args.num_runs,
                continue_timestamp=args.continue_timestamp,
                start_run_number=max_run,
                no_regenerate=use_no_regenerate,
                existing_answer_results=answer_results
            ))
        else:
            benchmark = BenchmarkV1(
                dataset_name=args.dataset,
                model_name=args.model,
                judge_model_name=args.judge_model,
                extra_body=extra_body,
                verbose_name=args.verbose_name
            )
            
            asyncio.run(benchmark.run_multiple_benchmarks(
                num_runs=args.num_runs,
                no_regenerate=args.no_regenerate
            ))

if __name__ == "__main__":
    main()
