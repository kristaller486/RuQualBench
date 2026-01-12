"""
Сервис оценки качества русского языка.

Использует существующую логику из benchmark/v1.py и benchmark/v2.py.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader
from litellm import acompletion

from benchmark.transport import create_transport
from benchmark.utils import extract_json_from_response, split_into_numbered_sentences
from server.models import (
    BatchItem,
    BatchItemResultV1,
    BatchItemResultV2,
    BatchSummary,
    EvaluateResponseV1,
    EvaluateResponseV2,
    MistakeV2,
)

logger = logging.getLogger(__name__)


class EvaluatorService:
    """
    Сервис для оценки качества русского языка в ответах LLM.
    
    Использует модель-судью для анализа текста и выявления ошибок.
    """
    
    def __init__(self):
        """Инициализирует сервис с настройками из переменных окружения."""
        # Настройки judge модели
        self.judge_model_name = os.getenv("JUDGE_MODEL_NAME")
        self.judge_api_base = os.getenv("JUDGE_MODEL_BASE_URL")
        self.judge_api_key = os.getenv("JUDGE_MODEL_API_KEY")
        self.judge_max_workers = int(os.getenv("JUDGE_MODEL_MAX_WORKERS", "10"))
        
        judge_extra_body_str = os.getenv("JUDGE_MODEL_EXTRA_BODY")
        self.judge_extra_body = json.loads(judge_extra_body_str) if judge_extra_body_str else None
        
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("RETRY_DELAY", "1.0"))
        
        # Jinja2 окружение для шаблонов
        self._jinja_env = Environment(loader=FileSystemLoader('prompts'))
        
        logger.info(f"EvaluatorService инициализирован: judge_model={self.judge_model_name}, max_workers={self.judge_max_workers}")
    
    def _normalize_explanations(self, explanations: List[Any]) -> List[str]:
        """Нормализует список объяснений - все элементы должны быть строками."""
        normalized = []
        for item in explanations:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(json.dumps(item, ensure_ascii=False))
            else:
                normalized.append(str(item))
        return normalized
    
    async def evaluate_v1(self, dialog: List[Dict[str, str]], answer: str) -> EvaluateResponseV1:
        """
        Оценивает ответ по методологии V1.
        
        Args:
            dialog: История диалога
            answer: Ответ модели для оценки
            
        Returns:
            Результат оценки с подсчётом ошибок по категориям
        """
        # Загружаем шаблоны
        system_template = self._jinja_env.get_template('judge_system.jinja2')
        user_template = self._jinja_env.get_template('judge_user.jinja2')
        system_prompt = system_template.render()
        
        # Подготавливаем данные для промпта
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
        
        # Выполняем запрос к judge модели с ретраями
        last_error = None
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.judge_model_name,
                    "messages": messages,
                    "api_base": self.judge_api_base,
                    "api_key": self.judge_api_key
                }
                if self.judge_extra_body:
                    kwargs["extra_body"] = self.judge_extra_body
                
                response = await acompletion(**kwargs)
                result = extract_json_from_response(response.choices[0].message.content)
                
                return EvaluateResponseV1(
                    critical_mistakes=result.get("critical_mistakes", 0),
                    mistakes=result.get("mistakes", 0),
                    additional_mistakes=result.get("additional_mistakes", 0),
                    explanation_critical_mistakes=self._normalize_explanations(result.get("explanation_critical_mistakes", [])),
                    explanation_mistakes=self._normalize_explanations(result.get("explanation_mistakes", [])),
                    explanation_additional_mistakes=self._normalize_explanations(result.get("explanation_additional_mistakes", []))
                )
            except Exception as e:
                last_error = e
                logger.warning(f"Ошибка при оценке V1 (попытка {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        logger.error(f"Ошибка при оценке V1 после {self.max_retries} попыток: {last_error}")
        raise last_error
    
    async def evaluate_v2(self, dialog: List[Dict[str, str]], answer: str) -> EvaluateResponseV2:
        """
        Оценивает ответ по методологии V2.
        
        Args:
            dialog: История диалога
            answer: Ответ модели для оценки
            
        Returns:
            Результат оценки с детальным списком ошибок
        """
        # Загружаем шаблоны
        system_template = self._jinja_env.get_template('judge_system_v2.jinja')
        user_template = self._jinja_env.get_template('judge_user_v2.jinja')
        system_prompt = system_template.render()
        
        # Подготавливаем данные для промпта
        history_for_judge = dialog[:-1] if len(dialog) > 1 else None
        user_prompt = dialog[-1]["content"]
        splitted_answer = split_into_numbered_sentences(answer)
        
        user_content = user_template.render(
            history=history_for_judge,
            prompt=user_prompt,
            answer=answer,
            splitted_answer=splitted_answer
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # Выполняем запрос к judge модели с ретраями
        last_error = None
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.judge_model_name,
                    "messages": messages,
                    "api_base": self.judge_api_base,
                    "api_key": self.judge_api_key
                }
                if self.judge_extra_body:
                    kwargs["extra_body"] = self.judge_extra_body
                
                response = await acompletion(**kwargs)
                raw_response_content = response.choices[0].message.content
                
                # Ожидаем список ошибок
                result_list = extract_json_from_response(raw_response_content)
                if not isinstance(result_list, list):
                    if isinstance(result_list, dict) and "items" in result_list:
                        result_list = result_list["items"]
                    else:
                        logger.warning(f"Judge V2 вернул не список: {type(result_list)}")
                        result_list = []
                
                # Подсчитываем ошибки по уровням
                mistakes_count = {"1": 0, "2": 0, "3": 0}
                mistakes = []
                
                for error in result_list:
                    level = error.get("level")
                    if level in [1, 2, 3]:
                        mistakes_count[str(level)] += 1
                    
                    mistakes.append(MistakeV2(
                        position=error.get("position", []),
                        level=error.get("level", 0),
                        type=error.get("type", "unknown"),
                        explanation=error.get("explanation", "")
                    ))
                
                return EvaluateResponseV2(
                    mistakes=mistakes,
                    mistakes_count=mistakes_count,
                    splitted_answer=splitted_answer
                )
            except Exception as e:
                last_error = e
                logger.warning(f"Ошибка при оценке V2 (попытка {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        logger.error(f"Ошибка при оценке V2 после {self.max_retries} попыток: {last_error}")
        raise last_error
    
    async def evaluate_batch_v1(
        self, 
        items: List[BatchItem],
        progress_callback: callable = None
    ) -> tuple[List[BatchItemResultV1], BatchSummary]:
        """
        Оценивает батч элементов по методологии V1.
        
        Args:
            items: Список элементов для оценки
            progress_callback: Callback для обновления прогресса
            
        Returns:
            Кортеж (результаты, сводка)
        """
        semaphore = asyncio.Semaphore(self.judge_max_workers)
        
        async def process_item(item: BatchItem) -> BatchItemResultV1:
            async with semaphore:
                try:
                    dialog = [{"role": m.role, "content": m.content} for m in item.dialog]
                    result = await self.evaluate_v1(dialog, item.answer)
                    
                    if progress_callback:
                        await progress_callback()
                    
                    return BatchItemResultV1(
                        id=item.id,
                        critical_mistakes=result.critical_mistakes,
                        mistakes=result.mistakes,
                        additional_mistakes=result.additional_mistakes,
                        explanation_critical_mistakes=result.explanation_critical_mistakes,
                        explanation_mistakes=result.explanation_mistakes,
                        explanation_additional_mistakes=result.explanation_additional_mistakes,
                        error=None
                    )
                except Exception as e:
                    logger.error(f"Ошибка при оценке элемента {item.id}: {e}")
                    
                    if progress_callback:
                        await progress_callback()
                    
                    return BatchItemResultV1(
                        id=item.id,
                        critical_mistakes=0,
                        mistakes=0,
                        additional_mistakes=0,
                        explanation_critical_mistakes=[],
                        explanation_mistakes=[],
                        explanation_additional_mistakes=[],
                        error=str(e)
                    )
        
        tasks = [process_item(item) for item in items]
        results = await asyncio.gather(*tasks)
        
        # Сортируем по порядку входных данных
        results_dict = {r.id: r for r in results}
        sorted_results = [results_dict[item.id] for item in items]
        
        # Формируем сводку
        successful = sum(1 for r in sorted_results if r.error is None)
        summary = BatchSummary(
            total=len(items),
            successful=successful,
            failed=len(items) - successful
        )
        
        return sorted_results, summary
    
    async def evaluate_batch_v2(
        self, 
        items: List[BatchItem],
        progress_callback: callable = None
    ) -> tuple[List[BatchItemResultV2], BatchSummary]:
        """
        Оценивает батч элементов по методологии V2.
        
        Args:
            items: Список элементов для оценки
            progress_callback: Callback для обновления прогресса
            
        Returns:
            Кортеж (результаты, сводка)
        """
        semaphore = asyncio.Semaphore(self.judge_max_workers)
        
        async def process_item(item: BatchItem) -> BatchItemResultV2:
            async with semaphore:
                try:
                    dialog = [{"role": m.role, "content": m.content} for m in item.dialog]
                    result = await self.evaluate_v2(dialog, item.answer)
                    
                    if progress_callback:
                        await progress_callback()
                    
                    return BatchItemResultV2(
                        id=item.id,
                        mistakes=result.mistakes,
                        mistakes_count=result.mistakes_count,
                        splitted_answer=result.splitted_answer,
                        error=None
                    )
                except Exception as e:
                    logger.error(f"Ошибка при оценке элемента {item.id}: {e}")
                    
                    if progress_callback:
                        await progress_callback()
                    
                    return BatchItemResultV2(
                        id=item.id,
                        mistakes=[],
                        mistakes_count={"1": 0, "2": 0, "3": 0},
                        splitted_answer="",
                        error=str(e)
                    )
        
        tasks = [process_item(item) for item in items]
        results = await asyncio.gather(*tasks)
        
        # Сортируем по порядку входных данных
        results_dict = {r.id: r for r in results}
        sorted_results = [results_dict[item.id] for item in items]
        
        # Формируем сводку
        successful = sum(1 for r in sorted_results if r.error is None)
        summary = BatchSummary(
            total=len(items),
            successful=successful,
            failed=len(items) - successful
        )
        
        return sorted_results, summary
