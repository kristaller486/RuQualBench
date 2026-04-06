import asyncio
import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

from benchmark.transport import Transport
from benchmark.utils import extract_json_from_response, split_into_numbered_sentences

logger = logging.getLogger(__name__)


def build_empty_mistakes_count() -> Dict[str, int]:
    return {"1": 0, "2": 0, "3": 0}


@dataclass
class V2JudgeResult:
    mistakes: List[Dict[str, Any]]
    mistakes_count: Dict[str, int]
    splitted_answer: str
    raw_judge_response_text: Optional[str] = None
    rendered_user_prompt: Optional[str] = None

    def to_dict(self, include_debug: bool = True) -> Dict[str, Any]:
        result = {
            "mistakes": self.mistakes,
            "mistakes_count": self.mistakes_count,
            "splitted_answer": self.splitted_answer,
        }
        if include_debug and self.raw_judge_response_text is not None:
            result["raw_judge_response_text"] = self.raw_judge_response_text
        if include_debug and self.rendered_user_prompt is not None:
            result["rendered_user_prompt"] = self.rendered_user_prompt
        return result


class V2Judge:
    def __init__(self, transport: Transport, debug_logs: int = 0):
        self.transport = self._clone_transport(transport)
        self.max_retries = max(1, self.transport.config.max_retries)
        self.retry_delay = self.transport.config.retry_delay
        self.transport.config.max_retries = 1
        self.debug_logs = debug_logs
        self._jinja_env = Environment(loader=FileSystemLoader('prompts'))
        self._system_template = self._jinja_env.get_template('judge_system_v2.jinja')
        self._user_template = self._jinja_env.get_template('judge_user_v2.jinja')

    @staticmethod
    def _clone_transport(transport: Transport) -> Transport:
        config = copy.deepcopy(transport.config)
        return type(transport)(config)

    async def evaluate(
        self,
        dialog: List[Dict[str, str]],
        answer: str,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> V2JudgeResult:
        if semaphore is None:
            return await self._evaluate_impl(dialog, answer)

        async with semaphore:
            return await self._evaluate_impl(dialog, answer)

    async def _evaluate_impl(self, dialog: List[Dict[str, str]], answer: str) -> V2JudgeResult:
        history_for_judge = dialog[:-1] if len(dialog) > 1 else None
        user_prompt = dialog[-1]["content"]
        splitted_answer = split_into_numbered_sentences(answer)
        system_prompt = self._system_template.render()
        user_content = self._user_template.render(
            history=history_for_judge,
            prompt=user_prompt,
            answer=answer,
            splitted_answer=splitted_answer,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            result_list: Optional[Any] = None
            try:
                raw_response_content = await self.transport.generate(messages)
                result_list = extract_json_from_response(raw_response_content)
                if not isinstance(result_list, list):
                    if isinstance(result_list, dict) and "items" in result_list:
                        result_list = result_list["items"]
                    else:
                        logger.warning(f"Judge v2 вернул не список: {type(result_list)}")
                        result_list = []

                normalized_result_list = result_list if isinstance(result_list, list) else []

                mistakes_count = build_empty_mistakes_count()
                mistakes = []
                for mistake in normalized_result_list:
                    level = mistake.get("level")
                    if level in [1, 2, 3]:
                        mistakes_count[str(level)] += 1
                    mistakes.append({
                        "position": mistake.get("position", []),
                        "level": mistake.get("level", 0),
                        "type": mistake.get("type", "unknown"),
                        "explanation": mistake.get("explanation", ""),
                    })

                return V2JudgeResult(
                    mistakes=mistakes,
                    mistakes_count=mistakes_count,
                    splitted_answer=splitted_answer,
                    raw_judge_response_text=raw_response_content if self.debug_logs >= 1 else None,
                    rendered_user_prompt=user_content if self.debug_logs >= 2 else None,
                )
            except Exception as exc:
                last_error = exc
                error_message = f"Ошибка при оценке v2 (попытка {attempt + 1}/{self.max_retries}): {exc}"
                if result_list is not None:
                    error_message += f"\nТип ответа: {type(result_list)}"
                    error_message += f"\nСодержимое (первые 500 симв.): {str(result_list)[:500]}"
                logger.warning(error_message)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"Не удалось оценить ответ судьей v2: {last_error}") from last_error
