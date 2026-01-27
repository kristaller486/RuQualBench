import asyncio
import json
import logging
import os
import statistics
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from tqdm.asyncio import tqdm as atqdm

from benchmark.utils import load_dataset, count_tokens, remove_think_tags
from benchmark.transport import (
    Transport, create_transport, GenerateRequest, GenerateResponse
)

logger = logging.getLogger(__name__)


class BenchmarkBase(ABC):
    def __init__(self, dataset_name: str, model_name: str = None, judge_model_name: str = None, 
                 extra_body: dict = None, verbose_name: str = None):
        self.dataset_name = dataset_name
        self.model_name = model_name or os.getenv("TEST_MODEL_NAME")
        self.judge_model_name = judge_model_name or os.getenv("JUDGE_MODEL_NAME")
        self.extra_body = extra_body
        self.verbose_name = verbose_name
        
        # Настройки из env (для обратной совместимости и логирования)
        self.test_api_base = os.getenv("TEST_MODEL_BASE_URL")
        self.test_api_key = os.getenv("TEST_MODEL_API_KEY")
        self.test_max_workers = int(os.getenv("TEST_MODEL_MAX_WORKERS", "10"))
        self.test_temperature = float(os.getenv("TEST_MODEL_TEMPERATURE", "1.0"))
        self.test_max_tokens = int(os.getenv("TEST_MODEL_MAX_TOKENS", "8192"))
        
        self.judge_api_base = os.getenv("JUDGE_MODEL_BASE_URL")
        self.judge_api_key = os.getenv("JUDGE_MODEL_API_KEY")
        self.judge_max_workers = int(os.getenv("JUDGE_MODEL_MAX_WORKERS", "10"))
        
        judge_extra_body_str = os.getenv("JUDGE_MODEL_EXTRA_BODY")
        self.judge_extra_body = json.loads(judge_extra_body_str) if judge_extra_body_str else None
        
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("RETRY_DELAY", "1.0"))
        
        self.logs_dir = Path("logs")  # Может быть переопределено в наследниках
        
        # Создаем транспорты
        self.test_transport = create_transport("TEST_MODEL")
        self.judge_transport = create_transport("JUDGE_MODEL")
        
        # Переопределяем model_name из транспорта если не задан явно
        if not model_name:
            self.model_name = self.test_transport.config.model
        else:
            self.test_transport.config.model = self.model_name

        if not judge_model_name:
            self.judge_model_name = self.judge_transport.config.model
        else:
            self.judge_transport.config.model = self.judge_model_name

        # Обновляем extra_body транспорта, если передан параметр --extra-body
        if self.extra_body:
            if self.test_transport.config.extra_body:
                self.test_transport.config.extra_body.update(self.extra_body)
            else:
                self.test_transport.config.extra_body = self.extra_body

    async def generate_answers_batch(self, dataset: List[List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        """Генерирует ответы для всего датасета через транспорт"""
        
        # Подготавливаем запросы
        requests = [
            GenerateRequest(
                id=i,
                messages=dialog,
                temperature=self.test_temperature,
                max_tokens=self.test_max_tokens
            )
            for i, dialog in enumerate(dataset)
        ]
        
        # Создаем progress bar
        pbar = atqdm(total=len(requests), desc="Генерация ответов")
        
        def progress_callback():
            pbar.update(1)
        
        # Вызываем транспорт
        responses = await self.test_transport.generate_batch(requests, progress_callback)
        pbar.close()
        
        # Преобразуем ответы в формат результатов
        results = []
        for response in responses:
            content = response.content
            
            # Обработка think tags если нужно
            if content and os.getenv("TEST_MODEL_EXCLUDE_THINK", "").lower() in ["true", "1", "yes"]:
                content = remove_think_tags(content)
            
            results.append({
                "dialog_id": response.id,
                "dialog": dataset[response.id],
                "answer": content,
                "tokens": count_tokens(content) if content else 0,
                "error": response.error
            })
        
        # Сортируем по id
        results.sort(key=lambda x: x["dialog_id"])
        return results

    async def generate_answer(self, messages: List[Dict[str, str]], semaphore: asyncio.Semaphore) -> str:
        """Генерирует ответ тестируемой модели асинхронно с ретраями (legacy метод)"""
        async with semaphore:
            content = await self.test_transport.generate(
                messages,
                temperature=self.test_temperature,
                max_tokens=self.test_max_tokens
            )
            
            if os.getenv("TEST_MODEL_EXCLUDE_THINK", "").lower() in ["true", "1", "yes"]:
                content = remove_think_tags(content)
            
            return content

    async def generate_dialog_answer(self, dialog: List[Dict[str, str]], dialog_id: int, 
                                     semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Генерирует ответ для диалога (legacy метод)"""
        try:
            answer = await self.generate_answer(dialog, semaphore)
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

    @abstractmethod
    async def judge_dialog_answer(self, answer_result: Dict[str, Any], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Оценивает готовый ответ диалога (реализуется в наследниках)"""
        pass

    @abstractmethod
    def calculate_summary(self, results: List[Dict[str, Any]], dataset_len: int) -> Dict[str, Any]:
        """Вычисляет статистику по результатам (реализуется в наследниках)"""
        pass

    async def run_single_benchmark(self, run_number: int = None, 
                                  existing_answer_results: List[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """Запускает один прогон бенчмарка"""
        run_prefix = f"Прогон {run_number}: " if run_number is not None else ""
        
        # Логируем информацию о транспортах
        test_transport_type = os.getenv("TEST_MODEL_TRANSPORT", "litellm")
        judge_transport_type = os.getenv("JUDGE_MODEL_TRANSPORT", "litellm")
        
        logger.info(f"{run_prefix}Тестируемая модель: {self.model_name} (transport={test_transport_type}, max_workers={self.test_max_workers}, temperature={self.test_temperature})")
        if self.extra_body:
            logger.info(f"{run_prefix}Test extra body: {self.extra_body}")
        logger.info(f"{run_prefix}Модель-оценщик: {self.judge_model_name} (transport={judge_transport_type}, max_workers={self.judge_max_workers})")
        if self.judge_extra_body:
            logger.info(f"{run_prefix}Judge extra body: {self.judge_extra_body}")
        logger.info(f"{run_prefix}Датасет: {self.dataset_name}")
        
        dataset = load_dataset(self.dataset_name)
        
        # Ограничиваем количество воркеров размером датасета
        actual_test_workers = min(self.test_max_workers, len(dataset))
        actual_judge_workers = min(self.judge_max_workers, len(dataset))
        
        judge_semaphore = asyncio.Semaphore(actual_judge_workers)
        
        # Этап 1: Генерация (или использование готовых ответов)
        if existing_answer_results:
            logger.info(f"{run_prefix}Используются существующие ответы ({len(existing_answer_results)} шт.)")
            answer_results = existing_answer_results
        else:
            logger.info(f"{run_prefix}Этап 1: Генерация ответов от тестируемой модели")
            
            # Используем batch метод транспорта
            if self.test_transport.is_batch_native:
                logger.info(f"{run_prefix}Используется нативный Batch API")
            
            answer_results = await self.generate_answers_batch(dataset)
        
        # Этап 2: Оценка
        logger.info(f"{run_prefix}Этап 2: Оценка ответов судьей")
        
        # Для судьи пока используем старый подход с semaphore
        # TODO: В будущем можно добавить batch для судьи
        judging_tasks = [
            self.judge_dialog_answer(answer_result, judge_semaphore)
            for answer_result in answer_results
        ]
        
        results = []
        for coro in atqdm.as_completed(judging_tasks, desc="Оценка ответов", total=len(judging_tasks)):
            result = await coro
            results.append(result)
        
        results.sort(key=lambda x: x["dialog_id"])
        
        summary = self.calculate_summary(results, len(dataset))
        
        config = {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "judge_model": self.judge_model_name,
            "test_max_workers": self.test_max_workers,
            "judge_max_workers": self.judge_max_workers,
            "test_transport": test_transport_type,
            "judge_transport": judge_transport_type
        }
        
        return results, summary, config

    async def run_multiple_benchmarks(self, num_runs: int, continue_timestamp: str = None, 
                                     start_run_number: int = 0, no_regenerate: bool = False,
                                     existing_answer_results: List[Dict[str, Any]] = None):
        """Запускает бенчмарк несколько раз"""
        self.logs_dir.mkdir(exist_ok=True)
        
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
        
        # Подготовка shared_answer_results для режима no_regenerate
        if no_regenerate:
            if existing_answer_results:
                logger.info("Используются ответы из run_1 существующей серии")
                shared_answer_results = existing_answer_results
            else:
                logger.info("")
                logger.info("="*70)
                logger.info("ГЕНЕРАЦИЯ ОТВЕТОВ (один раз для всех прогонов)")
                logger.info("="*70)
                
                # Запускаем генерацию без оценки (фиктивный прогон для получения ответов)
                # Но проще запустить полный прогон и взять ответы
                results, _, _ = await self.run_single_benchmark(run_number=None)
                
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
            
            current_answers = shared_answer_results if no_regenerate else None
            
            results, summary, config = await self.run_single_benchmark(
                run_number=run_num,
                existing_answer_results=current_answers
            )
            
            all_summaries.append(summary)
            
            log_filename = self.logs_dir / f"benchmark_{base_timestamp}_run_{run_num}_{self.dataset_name}.json"
            
            log_data = {
                "config": {
                    **config,
                    "timestamp": base_timestamp,
                    "run_number": run_num,
                    "total_runs": end_num,
                    "no_regenerate": no_regenerate
                },
                "results": results,
                "summary": summary
            }
            
            if self.verbose_name:
                log_data["config"]["verbose_name"] = self.verbose_name
            
            with open(log_filename, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Прогон {run_num}: Обработано {summary['successful_dialogs']}/{summary['total_dialogs']} диалогов")
            logger.info(f"Прогон {run_num}: Результаты сохранены в {log_filename.as_posix()}")
            
        if num_runs > 1:
            self.print_aggregated_stats(all_summaries, num_runs)

    def print_aggregated_stats(self, all_summaries: List[Dict[str, Any]], num_runs: int):
        """Выводит агрегированную статистику"""
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
