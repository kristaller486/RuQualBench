"""
Transport Layer для работы с различными LLM API.

Поддерживаемые транспорты:
- litellm: Универсальный клиент через LiteLLM
- google_genai: Прямой доступ к Google GenAI API
- google_genai_batch: Google GenAI Batch API (50% скидка, асинхронная обработка)
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TransportConfig:
    """Конфигурация транспорта"""
    model: str
    api_key: str
    base_url: Optional[str] = None
    temperature: float = 1.0
    max_tokens: int = 8192
    extra_body: Optional[Dict[str, Any]] = None
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class GenerateRequest:
    """Запрос на генерацию"""
    id: int
    messages: List[Dict[str, str]]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass
class GenerateResponse:
    """Ответ от генерации"""
    id: int
    content: Optional[str]
    error: Optional[str] = None


class Transport(ABC):
    """Абстрактный базовый класс для транспортов"""
    
    def __init__(self, config: TransportConfig):
        self.config = config
    
    @property
    def is_batch_native(self) -> bool:
        """Возвращает True, если транспорт использует нативный Batch API"""
        return False
    
    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]], 
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> str:
        """Генерирует ответ для одного запроса"""
        pass
    
    async def generate_batch(self, requests: List[GenerateRequest],
                             progress_callback: Optional[Callable[[], None]] = None) -> List[GenerateResponse]:
        """
        Генерирует ответы для пакета запросов.
        
        По умолчанию эмулирует батч через параллельные вызовы generate().
        Транспорты с нативным Batch API переопределяют этот метод.
        """
        semaphore = asyncio.Semaphore(10)  # Ограничение параллельных запросов
        
        async def process_request(request: GenerateRequest) -> GenerateResponse:
            async with semaphore:
                try:
                    content = await self.generate(
                        request.messages,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens
                    )
                    if progress_callback:
                        progress_callback()
                    return GenerateResponse(id=request.id, content=content)
                except Exception as e:
                    logger.error(f"Ошибка при генерации для запроса {request.id}: {e}")
                    if progress_callback:
                        progress_callback()
                    return GenerateResponse(id=request.id, content=None, error=str(e))
        
        tasks = [process_request(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        return list(responses)


class LiteLLMTransport(Transport):
    """Транспорт через LiteLLM"""
    
    def __init__(self, config: TransportConfig):
        super().__init__(config)
        # Импортируем здесь, чтобы не требовать litellm если не используется
        from litellm import acompletion
        self._acompletion = acompletion
    
    async def generate(self, messages: List[Dict[str, str]],
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> str:
        """Генерирует ответ через LiteLLM"""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                kwargs = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": temperature or self.config.temperature,
                    "max_tokens": max_tokens or self.config.max_tokens
                }
                
                if self.config.api_key:
                    kwargs["api_key"] = self.config.api_key
                if self.config.base_url:
                    kwargs["api_base"] = self.config.base_url
                if self.config.extra_body:
                    kwargs["extra_body"] = self.config.extra_body
                
                response = await self._acompletion(**kwargs)
                return response.choices[0].message.content
                
            except Exception as e:
                last_error = e
                logger.warning(f"LiteLLM ошибка (попытка {attempt+1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        raise last_error


class GoogleGenAITransport(Transport):
    """Транспорт через Google GenAI SDK (прямой доступ)"""
    
    def __init__(self, config: TransportConfig):
        super().__init__(config)
        # Импортируем здесь, чтобы не требовать google-genai если не используется
        from google import genai
        
        client_kwargs = {}
        if config.api_key:
            client_kwargs["api_key"] = config.api_key
        if config.base_url:
            client_kwargs["http_options"] = {"base_url": config.base_url}
        
        self._client = genai.Client(**client_kwargs)
    
    def _convert_messages_to_contents(self, messages: List[Dict[str, str]]) -> Tuple[List[Dict], Optional[str]]:
        """Конвертирует формат messages в формат Google GenAI contents"""
        contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_instruction = content
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            else:  # user
                contents.append({"role": "user", "parts": [{"text": content}]})
        
        return contents, system_instruction
    
    async def generate(self, messages: List[Dict[str, str]],
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> str:
        """Генерирует ответ через Google GenAI"""
        last_error = None
        contents, system_instruction = self._convert_messages_to_contents(messages)
        
        for attempt in range(self.config.max_retries):
            try:
                config = {
                    "temperature": temperature or self.config.temperature,
                    "max_output_tokens": max_tokens or self.config.max_tokens
                }
                
                if system_instruction:
                    config["system_instruction"] = system_instruction
                
                # Google GenAI SDK синхронный, оборачиваем в executor
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self._client.models.generate_content(
                        model=self.config.model,
                        contents=contents,
                        config=config
                    )
                )
                
                return response.text
                
            except Exception as e:
                last_error = e
                logger.warning(f"GoogleGenAI ошибка (попытка {attempt+1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        raise last_error


class GoogleGenAIBatchTransport(Transport):
    """
    Транспорт через Google GenAI Batch API.
    
    Особенности:
    - 50% скидка от стандартной стоимости
    - Асинхронная обработка (до 24 часов, обычно быстрее)
    - Подходит для больших объемов запросов
    """
    
    def __init__(self, config: TransportConfig):
        super().__init__(config)
        from google import genai
        from google.genai import types
        
        client_kwargs = {}
        if config.api_key:
            client_kwargs["api_key"] = config.api_key
        if config.base_url:
            client_kwargs["http_options"] = {"base_url": config.base_url}
        
        self._client = genai.Client(**client_kwargs)
        self._types = types
    
    @property
    def is_batch_native(self) -> bool:
        return True
    
    def _convert_messages_to_contents(self, messages: List[Dict[str, str]]) -> Tuple[List[Dict], Optional[str]]:
        """Конвертирует формат messages в формат Google GenAI contents"""
        contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_instruction = content
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            else:  # user
                contents.append({"role": "user", "parts": [{"text": content}]})
        
        return contents, system_instruction
    
    async def generate(self, messages: List[Dict[str, str]],
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> str:
        """
        Для одиночных запросов используем обычный API.
        Batch API имеет смысл только для пакетов.
        """
        # Fallback на обычный API для одиночных запросов
        contents, system_instruction = self._convert_messages_to_contents(messages)
        
        config = {
            "temperature": temperature or self.config.temperature,
            "max_output_tokens": max_tokens or self.config.max_tokens
        }
        
        if system_instruction:
            config["system_instruction"] = system_instruction
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=config
            )
        )
        
        return response.text
    
    async def generate_batch(self, requests: List[GenerateRequest],
                             progress_callback: Optional[Callable[[], None]] = None) -> List[GenerateResponse]:
        """
        Использует Google GenAI Batch API для обработки пакета запросов.
        """
        if not requests:
            return []
        
        logger.info(f"Создание Batch Job для {len(requests)} запросов...")
        
        # Подготавливаем inline requests
        inline_requests = []
        for req in requests:
            contents, system_instruction = self._convert_messages_to_contents(req.messages)
            
            request_config = {
                "temperature": req.temperature or self.config.temperature,
                "max_output_tokens": req.max_tokens or self.config.max_tokens
            }
            
            request_data = {
                "contents": contents
            }
            
            if system_instruction:
                request_data["config"] = {
                    "system_instruction": {"parts": [{"text": system_instruction}]},
                    **request_config
                }
            else:
                request_data["config"] = request_config
            
            inline_requests.append(request_data)
        
        # Создаем batch job
        loop = asyncio.get_event_loop()
        batch_job = await loop.run_in_executor(
            None,
            lambda: self._client.batches.create(
                model=self.config.model,
                src=inline_requests,
                config={"display_name": f"benchmark-batch-{int(time.time())}"}
            )
        )
        
        job_name = batch_job.name
        logger.info(f"Batch Job создан: {job_name}")
        
        # Ожидаем завершения
        completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}
        poll_interval = 10  # секунд
        
        while True:
            batch_job = await loop.run_in_executor(
                None,
                lambda: self._client.batches.get(name=job_name)
            )
            
            state = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)
            
            if state in completed_states:
                break
            
            logger.info(f"Batch Job статус: {state}, ожидание {poll_interval}с...")
            await asyncio.sleep(poll_interval)
        
        logger.info(f"Batch Job завершен со статусом: {state}")
        
        # Обрабатываем результаты
        responses = []
        
        if state == 'JOB_STATE_SUCCEEDED':
            if batch_job.dest and batch_job.dest.inlined_responses:
                for i, inline_response in enumerate(batch_job.dest.inlined_responses):
                    if progress_callback:
                        progress_callback()
                    
                    if inline_response.response:
                        try:
                            text = inline_response.response.text
                            responses.append(GenerateResponse(id=requests[i].id, content=text))
                        except Exception as e:
                            responses.append(GenerateResponse(
                                id=requests[i].id, 
                                content=None, 
                                error=f"Ошибка извлечения ответа: {e}"
                            ))
                    elif inline_response.error:
                        responses.append(GenerateResponse(
                            id=requests[i].id,
                            content=None,
                            error=str(inline_response.error)
                        ))
                    else:
                        responses.append(GenerateResponse(
                            id=requests[i].id,
                            content=None,
                            error="Пустой ответ"
                        ))
            else:
                # Результаты в файле
                logger.warning("Результаты в файле - требуется скачивание")
                for i, req in enumerate(requests):
                    responses.append(GenerateResponse(
                        id=req.id,
                        content=None,
                        error="Результаты в файле, не поддерживается"
                    ))
        else:
            # Job failed
            error_msg = f"Batch Job завершился с ошибкой: {state}"
            if hasattr(batch_job, 'error') and batch_job.error:
                error_msg += f" - {batch_job.error}"
            
            for req in requests:
                if progress_callback:
                    progress_callback()
                responses.append(GenerateResponse(id=req.id, content=None, error=error_msg))
        
        return responses


def create_transport(prefix: str) -> Transport:
    """
    Фабрика для создания транспорта на основе переменных окружения.
    
    Args:
        prefix: Префикс переменных окружения (TEST_MODEL или JUDGE_MODEL)
    
    Returns:
        Экземпляр Transport
    """
    transport_type = os.getenv(f"{prefix}_TRANSPORT", "litellm").lower()
    
    config = TransportConfig(
        model=os.getenv(f"{prefix}_NAME", ""),
        api_key=os.getenv(f"{prefix}_API_KEY", ""),
        base_url=os.getenv(f"{prefix}_BASE_URL"),
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "1.0")),
        max_tokens=int(os.getenv(f"{prefix}_MAX_TOKENS", "8192")),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        retry_delay=float(os.getenv("RETRY_DELAY", "1.0"))
    )
    
    # Парсим extra_body если есть
    extra_body_str = os.getenv(f"{prefix}_EXTRA_BODY")
    if extra_body_str:
        config.extra_body = json.loads(extra_body_str)
    
    if transport_type == "litellm":
        logger.info(f"Создание LiteLLM транспорта для {prefix}")
        return LiteLLMTransport(config)
    elif transport_type == "google_genai":
        logger.info(f"Создание GoogleGenAI транспорта для {prefix}")
        return GoogleGenAITransport(config)
    elif transport_type == "google_genai_batch":
        logger.info(f"Создание GoogleGenAI Batch транспорта для {prefix}")
        return GoogleGenAIBatchTransport(config)
    else:
        raise ValueError(f"Неизвестный тип транспорта: {transport_type}")
