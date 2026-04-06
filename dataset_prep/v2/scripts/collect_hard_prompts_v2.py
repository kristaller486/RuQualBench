import argparse
import asyncio
import copy
import csv
import hashlib
import json
import logging
import os
import random
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import litellm
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_prep.v2.paths import OUTPUTS_DIR, PROMPTS_DIR

from benchmark.judge_v2 import V2Judge, build_empty_mistakes_count
from benchmark.transport import (
    GoogleGenAIBatchTransport,
    GoogleGenAITransport,
    LiteLLMTransport,
    Transport,
    TransportConfig,
    create_transport,
)
from benchmark.utils import count_tokens, load_dataset, split_into_numbered_sentences

litellm.ssl_verify = False
load_dotenv(REPO_ROOT / ".env")

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

BUILTIN_DATASETS = {"debug", "lite", "base", "large"}


@dataclass(frozen=True)
class ModelSpec:
    model: str
    verbose_name: Optional[str] = None
    transport: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_retries: Optional[int] = None
    retry_delay: Optional[float] = None
    extra_body: Optional[Dict[str, Any]] = None

    @property
    def display_name(self) -> str:
        return self.verbose_name or self.model

    @property
    def cache_key(self) -> str:
        extra_body_json = json.dumps(self.extra_body, ensure_ascii=False, sort_keys=True) if self.extra_body else "{}"
        api_key_hash = hashlib.sha256((self.api_key or "").encode('utf-8')).hexdigest()[:12]
        return "::".join([
            self.transport or "",
            self.model,
            self.base_url or "",
            str(self.temperature) if self.temperature is not None else "",
            str(self.max_tokens) if self.max_tokens is not None else "",
            str(self.max_retries) if self.max_retries is not None else "",
            str(self.retry_delay) if self.retry_delay is not None else "",
            api_key_hash,
            extra_body_json,
        ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сбор hardest prompts для RuQualBench v2"
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        required=True,
        help="TOML-файл с пулом моделей и параметрами подключения"
    )
    parser.add_argument(
        "--dataset-source",
        dest="dataset_sources",
        action="append",
        default=[],
        help="Источник диалогов: builtin dataset (debug/lite/base/large) или путь до JSON"
    )
    parser.add_argument(
        "--datasets-file",
        type=Path,
        help="JSON-файл со списком источников датасетов"
    )
    parser.add_argument(
        "--sample-models",
        type=int,
        default=5,
        help="Сколько случайных моделей брать на каждый диалог"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Ограничить число диалогов после перемешивания"
    )
    parser.add_argument(
        "--head",
        type=int,
        help="Взять первые N диалогов до перемешивания"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для перемешивания датасетов и выбора моделей"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        help="Переопределить judge-модель из .env"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "hardest_prompts_v2_output",
        help="Директория для сохранения результатов"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Возобновить остановленный запуск из существующей директории"
    )
    parser.add_argument(
        "--debug-logs",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Уровень отладочных полей judge: 0/1/2"
    )
    parser.add_argument(
        "--generation-workers",
        type=int,
        help="Переопределить число параллельных генераций"
    )
    parser.add_argument(
        "--judge-workers",
        type=int,
        help="Переопределить число параллельных оценок"
    )
    return parser.parse_args()


def load_json_file(file_path: Path) -> Any:
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def load_toml_file(file_path: Path) -> Dict[str, Any]:
    with open(file_path, 'rb') as file:
        return tomllib.load(file)


def normalize_model(item: Any) -> ModelSpec:
    if isinstance(item, str):
        if not item.strip():
            raise ValueError("Название модели не может быть пустым")
        return ModelSpec(model=item.strip())

    if not isinstance(item, dict):
        raise ValueError(f"Некорректное описание модели: {item!r}")

    model_name = item.get("model")
    if not model_name or not isinstance(model_name, str):
        raise ValueError(f"У модели отсутствует строковое поле 'model': {item!r}")

    verbose_name = item.get("verbose_name")
    transport = item.get("transport")
    api_key = item.get("api_key")
    base_url = item.get("base_url")
    temperature = item.get("temperature")
    max_tokens = item.get("max_tokens")
    max_retries = item.get("max_retries")
    retry_delay = item.get("retry_delay")
    extra_body = item.get("extra_body")
    if transport is not None and not isinstance(transport, str):
        raise ValueError(f"Поле transport должно быть строкой: {item!r}")
    if api_key is not None and not isinstance(api_key, str):
        raise ValueError(f"Поле api_key должно быть строкой: {item!r}")
    if base_url is not None and not isinstance(base_url, str):
        raise ValueError(f"Поле base_url должно быть строкой: {item!r}")
    if temperature is not None and not isinstance(temperature, (int, float)):
        raise ValueError(f"Поле temperature должно быть числом: {item!r}")
    if max_tokens is not None and not isinstance(max_tokens, int):
        raise ValueError(f"Поле max_tokens должно быть целым числом: {item!r}")
    if max_retries is not None and not isinstance(max_retries, int):
        raise ValueError(f"Поле max_retries должно быть целым числом: {item!r}")
    if retry_delay is not None and not isinstance(retry_delay, (int, float)):
        raise ValueError(f"Поле retry_delay должно быть числом: {item!r}")
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError(f"Поле extra_body должно быть объектом: {item!r}")

    return ModelSpec(
        model=model_name,
        verbose_name=verbose_name,
        transport=transport,
        api_key=api_key,
        base_url=base_url,
        temperature=float(temperature) if temperature is not None else None,
        max_tokens=max_tokens,
        max_retries=max_retries,
        retry_delay=float(retry_delay) if retry_delay is not None else None,
        extra_body=extra_body,
    )


def load_models(models_file: Path) -> List[ModelSpec]:
    raw_data = load_toml_file(models_file)
    raw_models = raw_data.get("models")

    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("TOML-файл моделей должен содержать непустой список [[models]]")

    return [normalize_model(item) for item in raw_models]


def load_dataset_sources(args: argparse.Namespace) -> List[str]:
    sources = list(args.dataset_sources)
    if args.datasets_file:
        raw_data = load_json_file(args.datasets_file)
        if isinstance(raw_data, dict):
            raw_sources = raw_data.get("datasets")
        else:
            raw_sources = raw_data

        if not isinstance(raw_sources, list):
            raise ValueError("Файл источников датасетов должен содержать список")

        for item in raw_sources:
            if isinstance(item, str):
                sources.append(item)
            elif isinstance(item, dict) and isinstance(item.get("source"), str):
                sources.append(item["source"])
            else:
                raise ValueError(f"Некорректное описание источника датасета: {item!r}")

    if not sources:
        raise ValueError("Нужно указать хотя бы один источник датасета")

    return sources


def normalize_dialog(dialog: Any, source_name: str, dialog_index: int) -> List[Dict[str, str]]:
    if not isinstance(dialog, list) or not dialog:
        raise ValueError(f"Диалог #{dialog_index} из {source_name} должен быть непустым списком")

    normalized_dialog = []
    for turn_index, turn in enumerate(dialog):
        if not isinstance(turn, dict):
            raise ValueError(f"Сообщение #{turn_index} в диалоге #{dialog_index} из {source_name} должно быть объектом")

        role = turn.get("role")
        content = turn.get("content")
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Некорректная роль {role!r} в диалоге #{dialog_index} из {source_name}")
        if not isinstance(content, str):
            raise ValueError(f"Контент сообщения #{turn_index} в диалоге #{dialog_index} из {source_name} должен быть строкой")

        normalized_dialog.append({"role": role, "content": content})

    return normalized_dialog


def load_dialogs_from_source(source: str) -> List[Dict[str, Any]]:
    if source in BUILTIN_DATASETS:
        raw_dialogs = load_dataset(source)
        source_name = source
    else:
        source_path = Path(source)
        raw_data = load_json_file(source_path)
        if isinstance(raw_data, dict):
            raw_dialogs = raw_data.get("dialogs")
        else:
            raw_dialogs = raw_data
        source_name = source_path.as_posix()

    if not isinstance(raw_dialogs, list) or not raw_dialogs:
        raise ValueError(f"Источник {source_name} должен содержать непустой список диалогов")

    dialogs = []
    for dialog_index, dialog in enumerate(raw_dialogs):
        normalized_dialog = normalize_dialog(dialog, source_name, dialog_index)
        dialog_json = json.dumps(normalized_dialog, ensure_ascii=False, sort_keys=True)
        dialog_hash = hashlib.sha256(dialog_json.encode('utf-8')).hexdigest()
        dialogs.append({
            "source": source_name,
            "source_dialog_id": dialog_index,
            "dialog": normalized_dialog,
            "dialog_hash": dialog_hash,
        })
    return dialogs


def build_prompt_records(sources: List[str], seed: int, head: Optional[int], limit: Optional[int]) -> List[Dict[str, Any]]:
    if head is not None and head <= 0:
        raise ValueError("--head должен быть положительным числом")
    if limit is not None and limit <= 0:
        raise ValueError("--limit должен быть положительным числом")

    prompt_records: List[Dict[str, Any]] = []
    for source in sources:
        prompt_records.extend(load_dialogs_from_source(source))

    if head is not None:
        prompt_records = prompt_records[:head]

    rng = random.Random(seed)
    rng.shuffle(prompt_records)

    if limit is not None:
        prompt_records = prompt_records[:limit]

    for index, record in enumerate(prompt_records):
        record["prompt_id"] = f"prompt_{index:06d}_{record['dialog_hash'][:12]}"

    return prompt_records


def combine_extra_body(base_extra_body: Optional[Dict[str, Any]], override_extra_body: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if base_extra_body and override_extra_body:
        merged = dict(base_extra_body)
        merged.update(override_extra_body)
        return merged
    if override_extra_body is not None:
        return dict(override_extra_body)
    if base_extra_body is not None:
        return dict(base_extra_body)
    return None


def create_model_transport(config: TransportConfig, transport_type: str) -> Transport:
    normalized_transport_type = transport_type.lower()

    if normalized_transport_type == "litellm":
        return LiteLLMTransport(config)
    if normalized_transport_type == "google_genai":
        return GoogleGenAITransport(config)
    if normalized_transport_type == "google_genai_batch":
        return GoogleGenAIBatchTransport(config)

    raise ValueError(f"Неизвестный тип транспорта: {transport_type}")


def safe_count_message_tokens(model: str, messages: List[Dict[str, str]]) -> Optional[int]:
    try:
        token_counter = getattr(litellm, "token_counter")
        return int(token_counter(model=model, messages=messages))
    except Exception:
        return None


def safe_count_text_tokens(model: str, text: str) -> Optional[int]:
    try:
        token_counter = getattr(litellm, "token_counter")
        return int(token_counter(model=model, text=text, count_response_tokens=True))
    except Exception:
        return None


def safe_calculate_cost(model: str, messages: List[Dict[str, str]], completion: str) -> Optional[float]:
    try:
        completion_cost = getattr(litellm, "completion_cost")
        return float(completion_cost(
            model=model,
            messages=messages,
            completion=completion,
            call_type="acompletion",
        ))
    except Exception:
        return None


class HardPromptsCollector:
    def __init__(
        self,
        models: List[ModelSpec],
        prompt_records: List[Dict[str, Any]],
        sample_models: int,
        seed: int,
        judge_model_name: Optional[str],
        debug_logs: int,
        generation_workers: Optional[int],
        judge_workers: Optional[int],
        initial_completed_results: Optional[List[Dict[str, Any]]] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        if len(models) < sample_models:
            raise ValueError(
                f"Недостаточно моделей в пуле: требуется {sample_models}, доступно {len(models)}"
            )
        if sample_models <= 0:
            raise ValueError("--sample-models должен быть положительным числом")

        self.models = models
        self.prompt_records = prompt_records
        self.sample_models = sample_models
        self.rng = random.Random(seed)
        self.base_test_transport = create_transport("TEST_MODEL")
        self.base_judge_transport = create_transport("JUDGE_MODEL")
        if judge_model_name:
            self.base_judge_transport.config.model = judge_model_name
        self.transport_cache: Dict[str, Transport] = {}
        self.output_debug_logs = debug_logs
        self.judge = V2Judge(self.base_judge_transport, debug_logs=max(debug_logs, 1))
        self.generation_workers = generation_workers or max(1, int(os.getenv("TEST_MODEL_MAX_WORKERS", "10")))
        self.judge_workers = judge_workers or max(1, int(os.getenv("JUDGE_MODEL_MAX_WORKERS", "10")))
        self._jinja_env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)))
        self._judge_system_template = self._jinja_env.get_template('judge_system_v2.jinja')
        self._judge_user_template = self._jinja_env.get_template('judge_user_v2.jinja')
        if self.generation_workers <= 0:
            raise ValueError("--generation-workers должен быть положительным числом")
        if self.judge_workers <= 0:
            raise ValueError("--judge-workers должен быть положительным числом")
        self.generation_semaphore = asyncio.Semaphore(self.generation_workers)
        self.judge_semaphore = asyncio.Semaphore(self.judge_workers)
        for prompt_record in self.prompt_records:
            prompt_record["sampled_models"] = self.rng.sample(self.models, self.sample_models)
        self.initial_completed_results = list(initial_completed_results or [])
        self.checkpoint_dir = checkpoint_dir

    def get_transport_for_model(self, model_spec: ModelSpec) -> Transport:
        if model_spec.cache_key in self.transport_cache:
            return self.transport_cache[model_spec.cache_key]

        config = copy.deepcopy(self.base_test_transport.config)
        config.model = model_spec.model
        if model_spec.api_key is not None:
            config.api_key = model_spec.api_key
        if model_spec.base_url is not None:
            config.base_url = model_spec.base_url
        if model_spec.temperature is not None:
            config.temperature = model_spec.temperature
        if model_spec.max_tokens is not None:
            config.max_tokens = model_spec.max_tokens
        if model_spec.max_retries is not None:
            config.max_retries = model_spec.max_retries
        if model_spec.retry_delay is not None:
            config.retry_delay = model_spec.retry_delay
        config.extra_body = combine_extra_body(config.extra_body, model_spec.extra_body)
        transport_type = model_spec.transport or os.getenv("TEST_MODEL_TRANSPORT", "litellm")
        transport = create_model_transport(config, transport_type)
        self.transport_cache[model_spec.cache_key] = transport
        return transport

    def build_judge_messages(self, dialog: List[Dict[str, str]], answer: str) -> List[Dict[str, str]]:
        history_for_judge = dialog[:-1] if len(dialog) > 1 else None
        user_prompt = dialog[-1]["content"]
        splitted_answer = split_into_numbered_sentences(answer)
        system_prompt = self._judge_system_template.render()
        user_content = self._judge_user_template.render(
            history=history_for_judge,
            prompt=user_prompt,
            answer=answer,
            splitted_answer=splitted_answer,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    async def generate_litellm_answer(self, transport: LiteLLMTransport, dialog: List[Dict[str, str]]) -> Dict[str, Any]:
        last_error: Optional[Exception] = None

        for attempt in range(transport.config.max_retries):
            try:
                kwargs = {
                    "model": transport.config.model,
                    "messages": dialog,
                    "temperature": transport.config.temperature,
                    "max_tokens": transport.config.max_tokens,
                }
                if transport.config.api_key:
                    kwargs["api_key"] = transport.config.api_key
                if transport.config.base_url:
                    kwargs["api_base"] = transport.config.base_url
                if transport.config.extra_body:
                    kwargs["extra_body"] = transport.config.extra_body

                response = await transport._acompletion(**kwargs)
                choices = getattr(response, "choices", None) or []
                first_choice = choices[0] if choices else None
                response_message = getattr(first_choice, "message", None)
                answer = getattr(response_message, "content", None) or ""
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                cost_usd = safe_calculate_cost(transport.config.model, dialog, answer)

                resolved_completion_tokens = int(completion_tokens) if completion_tokens is not None else count_tokens(answer)
                return {
                    "answer": answer,
                    "tokens": resolved_completion_tokens,
                    "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else safe_count_message_tokens(transport.config.model, dialog),
                    "completion_tokens": int(completion_tokens) if completion_tokens is not None else safe_count_text_tokens(transport.config.model, answer),
                    "cost_usd": cost_usd,
                }
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LiteLLM ошибка (попытка %s/%s): %s",
                    attempt + 1,
                    transport.config.max_retries,
                    exc,
                )
                if attempt < transport.config.max_retries - 1:
                    await asyncio.sleep(transport.config.retry_delay * (attempt + 1))

        raise RuntimeError(f"Не удалось сгенерировать ответ через LiteLLM: {last_error}") from last_error

    async def generate_answer(self, model_spec: ModelSpec, dialog: List[Dict[str, str]]) -> Dict[str, Any]:
        transport = self.get_transport_for_model(model_spec)
        async with self.generation_semaphore:
            if isinstance(transport, LiteLLMTransport):
                return await self.generate_litellm_answer(transport, dialog)

            answer = await transport.generate(dialog)

        tokens = count_tokens(answer) if answer else 0
        return {
            "answer": answer,
            "tokens": tokens,
            "prompt_tokens": None,
            "completion_tokens": None,
            "cost_usd": None,
        }

    async def process_model_result(self, prompt_record: Dict[str, Any], model_spec: ModelSpec) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "model": model_spec.model,
            "verbose_name": model_spec.verbose_name,
            "display_name": model_spec.display_name,
            "transport": model_spec.transport,
            "base_url": model_spec.base_url,
            "model_extra_body": model_spec.extra_body,
            "generation_error": None,
            "judge_error": None,
            "answer": None,
            "tokens": 0,
            "generation_prompt_tokens": None,
            "generation_completion_tokens": None,
            "generation_cost_usd": None,
            "judge_prompt_tokens": None,
            "judge_completion_tokens": None,
            "judge_cost_usd": None,
            "total_cost_usd": None,
            "mistakes": [],
            "mistakes_count": build_empty_mistakes_count(),
            "splitted_answer": "",
            "total_mistakes": 0,
            "mistakes_per_1000_tokens": 0.0,
        }

        try:
            generated = await self.generate_answer(model_spec, prompt_record["dialog"])
            result["answer"] = generated["answer"]
            result["tokens"] = generated["tokens"]
            result["generation_prompt_tokens"] = generated["prompt_tokens"]
            result["generation_completion_tokens"] = generated["completion_tokens"]
            result["generation_cost_usd"] = generated["cost_usd"]
        except Exception as exc:
            result["generation_error"] = str(exc)
            return result

        try:
            judged = await self.judge.evaluate(prompt_record["dialog"], result["answer"], self.judge_semaphore)
            judged_dict = judged.to_dict()
            result["mistakes"] = judged_dict["mistakes"]
            result["mistakes_count"] = judged_dict["mistakes_count"]
            result["splitted_answer"] = judged_dict["splitted_answer"]
            if self.output_debug_logs >= 1 and "raw_judge_response_text" in judged_dict:
                result["raw_judge_response_text"] = judged_dict["raw_judge_response_text"]
            if self.output_debug_logs >= 2 and "rendered_user_prompt" in judged_dict:
                result["rendered_user_prompt"] = judged_dict["rendered_user_prompt"]

            raw_judge_response_text = judged.raw_judge_response_text or ""
            if isinstance(self.judge.transport, LiteLLMTransport):
                judge_messages = self.build_judge_messages(prompt_record["dialog"], result["answer"])
                result["judge_prompt_tokens"] = safe_count_message_tokens(self.judge.transport.config.model, judge_messages)
                result["judge_completion_tokens"] = safe_count_text_tokens(self.judge.transport.config.model, raw_judge_response_text)
                result["judge_cost_usd"] = safe_calculate_cost(
                    self.judge.transport.config.model,
                    judge_messages,
                    raw_judge_response_text,
                )
        except Exception as exc:
            result["judge_error"] = str(exc)
            return result

        total_mistakes = sum(result["mistakes_count"].values())
        result["total_mistakes"] = total_mistakes
        result["mistakes_per_1000_tokens"] = round((total_mistakes / result["tokens"] * 1000), 4) if result["tokens"] > 0 else 0.0
        if result["generation_cost_usd"] is not None or result["judge_cost_usd"] is not None:
            result["total_cost_usd"] = round((result["generation_cost_usd"] or 0.0) + (result["judge_cost_usd"] or 0.0), 8)
        return result

    async def process_prompt(self, prompt_record: Dict[str, Any]) -> Dict[str, Any]:
        sampled_models = prompt_record["sampled_models"]
        model_results = await asyncio.gather(
            *(self.process_model_result(prompt_record, model_spec) for model_spec in sampled_models)
        )

        successful_results = [
            item for item in model_results
            if item["generation_error"] is None and item["judge_error"] is None
        ]

        aggregate = self.aggregate_prompt_results(successful_results)
        return {
            "prompt_id": prompt_record["prompt_id"],
            "source": prompt_record["source"],
            "source_dialog_id": prompt_record["source_dialog_id"],
            "dialog_hash": prompt_record["dialog_hash"],
            "dialog": prompt_record["dialog"],
            "sampled_models": [
                {
                    "model": model_spec.model,
                    "verbose_name": model_spec.verbose_name,
                    "display_name": model_spec.display_name,
                    "transport": model_spec.transport,
                    "base_url": model_spec.base_url,
                    "extra_body": model_spec.extra_body,
                }
                for model_spec in sampled_models
            ],
            "successful_models_count": len(successful_results),
            "failed_models_count": len(model_results) - len(successful_results),
            "aggregate": aggregate,
            "model_results": model_results,
        }

    @staticmethod
    def aggregate_prompt_results(successful_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not successful_results:
            return {
                "avg_level_1": None,
                "avg_level_2": None,
                "avg_level_3": None,
                "avg_total": None,
                "avg_tokens": None,
                "avg_total_per_1000_tokens": None,
                "sum_level_1": 0,
                "sum_level_2": 0,
                "sum_level_3": 0,
                "sum_total": 0,
                "sum_tokens": 0,
                "avg_generation_cost_usd": None,
                "avg_judge_cost_usd": None,
                "avg_total_cost_usd": None,
                "sum_generation_cost_usd": 0.0,
                "sum_judge_cost_usd": 0.0,
                "sum_total_cost_usd": 0.0,
            }

        sum_level_1 = sum(item["mistakes_count"]["1"] for item in successful_results)
        sum_level_2 = sum(item["mistakes_count"]["2"] for item in successful_results)
        sum_level_3 = sum(item["mistakes_count"]["3"] for item in successful_results)
        sum_total = sum(item["total_mistakes"] for item in successful_results)
        sum_tokens = sum(item["tokens"] for item in successful_results)
        successful_count = len(successful_results)
        generation_costs = [item["generation_cost_usd"] for item in successful_results if item["generation_cost_usd"] is not None]
        judge_costs = [item["judge_cost_usd"] for item in successful_results if item["judge_cost_usd"] is not None]
        total_costs = [item["total_cost_usd"] for item in successful_results if item["total_cost_usd"] is not None]
        sum_generation_cost = sum(generation_costs)
        sum_judge_cost = sum(judge_costs)
        sum_total_cost = sum(total_costs)

        return {
            "avg_level_1": round(sum_level_1 / successful_count, 4),
            "avg_level_2": round(sum_level_2 / successful_count, 4),
            "avg_level_3": round(sum_level_3 / successful_count, 4),
            "avg_total": round(sum_total / successful_count, 4),
            "avg_tokens": round(sum_tokens / successful_count, 4),
            "avg_total_per_1000_tokens": round((sum_total / sum_tokens * 1000), 4) if sum_tokens > 0 else 0.0,
            "sum_level_1": sum_level_1,
            "sum_level_2": sum_level_2,
            "sum_level_3": sum_level_3,
            "sum_total": sum_total,
            "sum_tokens": sum_tokens,
            "avg_generation_cost_usd": round(sum_generation_cost / len(generation_costs), 8) if generation_costs else None,
            "avg_judge_cost_usd": round(sum_judge_cost / len(judge_costs), 8) if judge_costs else None,
            "avg_total_cost_usd": round(sum_total_cost / len(total_costs), 8) if total_costs else None,
            "sum_generation_cost_usd": round(sum_generation_cost, 8),
            "sum_judge_cost_usd": round(sum_judge_cost, 8),
            "sum_total_cost_usd": round(sum_total_cost, 8),
        }

    async def run(self) -> List[Dict[str, Any]]:
        total_dialogs = len(self.initial_completed_results) + len(self.prompt_records)
        logger.info("Старт обработки %s диалогов", total_dialogs)

        results = list(self.initial_completed_results)
        observed_generation_cost, observed_judge_cost = calculate_observed_costs(results)
        progress = LiveProgressDisplay(
            total_dialogs=total_dialogs,
            processed_dialogs=len(results),
            observed_total_cost_usd=observed_generation_cost + observed_judge_cost,
        )
        progress.render()

        tasks = [asyncio.create_task(self.process_prompt(prompt_record)) for prompt_record in self.prompt_records]
        try:
            for completed_task in asyncio.as_completed(tasks):
                result = await completed_task
                results.append(result)
                generation_delta, judge_delta = calculate_observed_costs([result])
                observed_generation_cost += generation_delta
                observed_judge_cost += judge_delta
                progress.update(
                    processed_dialogs=len(results),
                    observed_total_cost_usd=observed_generation_cost + observed_judge_cost,
                )
                if self.checkpoint_dir is not None:
                    append_jsonl(self.checkpoint_dir / "checkpoint_results.jsonl", result)
        finally:
            progress.close()

        results.sort(
            key=lambda item: (
                item["aggregate"]["avg_total_per_1000_tokens"] is None,
                -(item["aggregate"]["avg_total_per_1000_tokens"] or -1),
                -(item["aggregate"]["avg_total"] or -1),
                item["prompt_id"],
            )
        )
        return results


def build_csv_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for rank, item in enumerate(results, start=1):
        dialog = item["dialog"]
        last_user_prompt = dialog[-1]["content"] if dialog else ""
        preview = last_user_prompt.replace("\n", " ")[:160]
        aggregate = item["aggregate"]
        rows.append({
            "rank": rank,
            "prompt_id": item["prompt_id"],
            "source": item["source"],
            "source_dialog_id": item["source_dialog_id"],
            "successful_models_count": item["successful_models_count"],
            "failed_models_count": item["failed_models_count"],
            "avg_level_1": aggregate["avg_level_1"],
            "avg_level_2": aggregate["avg_level_2"],
            "avg_level_3": aggregate["avg_level_3"],
            "avg_total": aggregate["avg_total"],
            "avg_tokens": aggregate["avg_tokens"],
            "avg_total_per_1000_tokens": aggregate["avg_total_per_1000_tokens"],
            "last_user_message_preview": preview,
        })
    return rows


def write_json(file_path: Path, payload: Any) -> None:
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_json_atomic(file_path: Path, payload: Any) -> None:
    temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")
    with open(temp_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, file_path)


def append_jsonl(file_path: Path, payload: Any) -> None:
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(payload, ensure_ascii=False))
        file.write("\n")


def load_jsonl(file_path: Path) -> List[Any]:
    if not file_path.exists():
        return []

    items: List[Any] = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(
                    "Пропускаю битую строку checkpoint_results.jsonl на строке %s",
                    line_number,
                )
    return items


def build_models_fingerprint(models: List[ModelSpec]) -> List[str]:
    return [model.cache_key for model in models]


def build_run_fingerprint(
    models: List[ModelSpec],
    dataset_sources: List[str],
    seed: int,
    head: Optional[int],
    limit: Optional[int],
    sample_models: int,
    judge_model: str,
) -> str:
    payload = {
        "models": build_models_fingerprint(models),
        "dataset_sources": dataset_sources,
        "seed": seed,
        "head": head,
        "limit": limit,
        "sample_models": sample_models,
        "judge_model": judge_model,
    }
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload_json.encode('utf-8')).hexdigest()


def build_checkpoint_manifest(
    *,
    timestamp: str,
    run_fingerprint: str,
    models_file: Path,
    dataset_sources: List[str],
    seed: int,
    head: Optional[int],
    limit: Optional[int],
    sample_models: int,
    judge_model: str,
    generation_workers: int,
    judge_workers: int,
    debug_logs: int,
    total_prompts: int,
    completed_prompt_ids: List[str],
    observed_generation_cost_usd: float,
    observed_judge_cost_usd: float,
    updated_at: str,
    status: str,
) -> Dict[str, Any]:
    return {
        "version": 1,
        "timestamp": timestamp,
        "run_fingerprint": run_fingerprint,
        "models_file": models_file.as_posix(),
        "dataset_sources": dataset_sources,
        "seed": seed,
        "head": head,
        "limit": limit,
        "sample_models": sample_models,
        "judge_model": judge_model,
        "generation_workers": generation_workers,
        "judge_workers": judge_workers,
        "debug_logs": debug_logs,
        "total_prompts": total_prompts,
        "completed_prompt_ids": completed_prompt_ids,
        "processed_prompts": len(completed_prompt_ids),
        "observed_generation_cost_usd": round(observed_generation_cost_usd, 8),
        "observed_judge_cost_usd": round(observed_judge_cost_usd, 8),
        "observed_total_cost_usd": round(observed_generation_cost_usd + observed_judge_cost_usd, 8),
        "updated_at": updated_at,
        "status": status,
    }


def calculate_observed_costs(results: List[Dict[str, Any]]) -> Tuple[float, float]:
    generation_cost = 0.0
    judge_cost = 0.0
    for item in results:
        for model_result in item.get("model_results", []):
            generation_cost += float(model_result.get("generation_cost_usd") or 0.0)
            judge_cost += float(model_result.get("judge_cost_usd") or 0.0)
    return generation_cost, judge_cost


def load_resume_state(run_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    manifest_path = run_dir / "checkpoint_manifest.json"
    results_path = run_dir / "checkpoint_results.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Не найден manifest для возобновления: {manifest_path}")

    manifest = load_json_file(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint_manifest.json должен содержать объект")

    loaded_results = load_jsonl(results_path)
    if any(not isinstance(item, dict) for item in loaded_results):
        raise ValueError("checkpoint_results.jsonl содержит некорректные записи")

    return manifest, loaded_results


class LiveProgressDisplay:
    def __init__(self, total_dialogs: int, processed_dialogs: int = 0, observed_total_cost_usd: float = 0.0):
        self.total_dialogs = total_dialogs
        self.processed_dialogs = processed_dialogs
        self.observed_total_cost_usd = observed_total_cost_usd
        self._last_length = 0
        self._enabled = sys.stderr.isatty()

    def update(self, processed_dialogs: int, observed_total_cost_usd: float) -> None:
        self.processed_dialogs = processed_dialogs
        self.observed_total_cost_usd = observed_total_cost_usd
        self.render()

    def render(self) -> None:
        message = (
            f"Обработано диалогов: {self.processed_dialogs}/{self.total_dialogs} | "
            f"Потрачено: ${self.observed_total_cost_usd:.6f}"
        )
        if self._enabled:
            padded_message = message.ljust(self._last_length)
            sys.stderr.write(f"\r{padded_message}")
            sys.stderr.flush()
            self._last_length = max(self._last_length, len(message))
        else:
            logger.info(message)

    def close(self) -> None:
        if self._enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()


def write_csv(file_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        rows = [{
            "rank": "",
            "prompt_id": "",
            "source": "",
            "source_dialog_id": "",
            "successful_models_count": "",
            "failed_models_count": "",
            "avg_level_1": "",
            "avg_level_2": "",
            "avg_level_3": "",
            "avg_total": "",
            "avg_tokens": "",
            "avg_total_per_1000_tokens": "",
            "last_user_message_preview": "",
        }]

    with open(file_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_run_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompts_with_success = sum(1 for item in results if item["successful_models_count"] > 0)
    prompts_without_success = len(results) - prompts_with_success
    total_successful_models = sum(item["successful_models_count"] for item in results)
    total_failed_models = sum(item["failed_models_count"] for item in results)
    total_generation_cost, total_judge_cost = calculate_observed_costs(results)
    top_prompt_ids = [item["prompt_id"] for item in results[:10]]

    return {
        "total_prompts": len(results),
        "prompts_with_success": prompts_with_success,
        "prompts_without_success": prompts_without_success,
        "total_successful_model_runs": total_successful_models,
        "total_failed_model_runs": total_failed_models,
        "total_generation_cost_usd": round(total_generation_cost, 8),
        "total_judge_cost_usd": round(total_judge_cost, 8),
        "total_cost_usd": round(total_generation_cost + total_judge_cost, 8),
        "top_10_prompt_ids": top_prompt_ids,
    }


async def async_main(args: argparse.Namespace) -> None:
    models = load_models(args.models_file)
    dataset_sources = load_dataset_sources(args)
    prompt_records = build_prompt_records(dataset_sources, seed=args.seed, head=args.head, limit=args.limit)

    if args.resume:
        run_dir = args.resume
        if not run_dir.exists():
            raise FileNotFoundError(f"Директория для возобновления не найдена: {run_dir}")
        timestamp = run_dir.name
        manifest, loaded_results = load_resume_state(run_dir)
        completed_prompt_ids: Set[str] = {item["prompt_id"] for item in loaded_results}
        logger.info("Возобновление из %s", run_dir)
        logger.info("Найдено %s завершенных диалогов", len(completed_prompt_ids))
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = args.output_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = None
        loaded_results = []
        completed_prompt_ids = set()

    collector = HardPromptsCollector(
        models=models,
        prompt_records=prompt_records,
        sample_models=args.sample_models,
        seed=args.seed,
        judge_model_name=args.judge_model,
        debug_logs=args.debug_logs,
        generation_workers=args.generation_workers,
        judge_workers=args.judge_workers,
        initial_completed_results=loaded_results,
        checkpoint_dir=run_dir,
    )

    effective_judge_model = collector.base_judge_transport.config.model
    run_fingerprint = build_run_fingerprint(
        models=models,
        dataset_sources=dataset_sources,
        seed=args.seed,
        head=args.head,
        limit=args.limit,
        sample_models=args.sample_models,
        judge_model=effective_judge_model,
    )

    if manifest is not None and manifest.get("run_fingerprint") != run_fingerprint:
        raise ValueError("Параметры запуска не совпадают с checkpoint, возобновление невозможно")

    collector.prompt_records = [
        prompt_record
        for prompt_record in collector.prompt_records
        if prompt_record["prompt_id"] not in completed_prompt_ids
    ]

    observed_generation_cost, observed_judge_cost = calculate_observed_costs(loaded_results)
    write_json_atomic(
        run_dir / "checkpoint_manifest.json",
        build_checkpoint_manifest(
            timestamp=timestamp,
            run_fingerprint=run_fingerprint,
            models_file=args.models_file,
            dataset_sources=dataset_sources,
            seed=args.seed,
            head=args.head,
            limit=args.limit,
            sample_models=args.sample_models,
            judge_model=effective_judge_model,
            generation_workers=collector.generation_workers,
            judge_workers=collector.judge_workers,
            debug_logs=args.debug_logs,
            total_prompts=len(prompt_records),
            completed_prompt_ids=sorted(completed_prompt_ids),
            observed_generation_cost_usd=observed_generation_cost,
            observed_judge_cost_usd=observed_judge_cost,
            updated_at=datetime.now().isoformat(timespec="seconds"),
            status="running",
        ),
    )

    results = await collector.run()

    metadata = {
        "timestamp": timestamp,
        "seed": args.seed,
        "head": args.head,
        "limit": args.limit,
        "sample_models": args.sample_models,
        "judge_model": effective_judge_model,
        "models_file": args.models_file.as_posix(),
        "dataset_sources": dataset_sources,
        "processed_prompts": len(results),
        "generation_workers": collector.generation_workers,
        "judge_workers": collector.judge_workers,
        "debug_logs": args.debug_logs,
        "resumed_from": args.resume.as_posix() if args.resume else None,
    }
    run_summary = build_run_summary(results)
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "run_summary.json", run_summary)
    write_json(run_dir / "prompt_results.json", results)
    write_csv(run_dir / "hardest_prompts.csv", build_csv_rows(results))

    final_generation_cost, final_judge_cost = calculate_observed_costs(results)
    write_json_atomic(
        run_dir / "checkpoint_manifest.json",
        build_checkpoint_manifest(
            timestamp=timestamp,
            run_fingerprint=run_fingerprint,
            models_file=args.models_file,
            dataset_sources=dataset_sources,
            seed=args.seed,
            head=args.head,
            limit=args.limit,
            sample_models=args.sample_models,
            judge_model=effective_judge_model,
            generation_workers=collector.generation_workers,
            judge_workers=collector.judge_workers,
            debug_logs=args.debug_logs,
            total_prompts=len(prompt_records),
            completed_prompt_ids=sorted(item["prompt_id"] for item in results),
            observed_generation_cost_usd=final_generation_cost,
            observed_judge_cost_usd=final_judge_cost,
            updated_at=datetime.now().isoformat(timespec="seconds"),
            status="completed",
        ),
    )

    logger.info(
        "Стоимость прогона: generation=$%.6f, judge=$%.6f, total=$%.6f",
        run_summary["total_generation_cost_usd"],
        run_summary["total_judge_cost_usd"],
        run_summary["total_cost_usd"],
    )
    logger.info("Готово. Результаты сохранены в %s", run_dir)


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
