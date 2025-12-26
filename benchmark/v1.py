import asyncio
import json
import logging
from typing import List, Dict, Any

from jinja2 import Environment, FileSystemLoader
from litellm import acompletion

from benchmark.base import BenchmarkBase
from benchmark.utils import extract_json_from_response

logger = logging.getLogger(__name__)

class BenchmarkV1(BenchmarkBase):
    async def judge_dialog_answer(self, answer_result: Dict[str, Any], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Оценивает готовый ответ диалога (V1)"""
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
            judge_result = await self._judge_answer(
                answer_result["dialog"],
                answer_result["answer"],
                semaphore
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

    async def _judge_answer(self, dialog: List[Dict[str, str]], answer: str, semaphore: asyncio.Semaphore) -> Dict[str, int]:
        """Внутренний метод оценки ответа с помощью модели-judge"""
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
                    logger.debug('Делаем запрос судьей...')
                    result = extract_json_from_response(response.choices[0].message.content)
                    
                    # Нормализуем explanation_* - все элементы должны быть строками
                    def normalize_explanations(explanations):
                        normalized = []
                        for item in explanations:
                            if isinstance(item, str):
                                normalized.append(item)
                            elif isinstance(item, dict):
                                normalized.append(json.dumps(item, ensure_ascii=False))
                            else:
                                normalized.append(str(item))
                        return normalized
                    
                    return {
                        "critical_mistakes": result.get("critical_mistakes", 0),
                        "mistakes": result.get("mistakes", 0),
                        "additional_mistakes": result.get("additional_mistakes", 0),
                        "explanation_critical_mistakes": normalize_explanations(result.get("explanation_critical_mistakes", [])),
                        "explanation_mistakes": normalize_explanations(result.get("explanation_mistakes", [])),
                        "explanation_additional_mistakes": normalize_explanations(result.get("explanation_additional_mistakes", []))
                    }
                except Exception as e:
                    last_error = e
                    logger.warning(f"Ошибка при оценке (попытка {attempt+1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    else:
                        logger.error(f"Ошибка при оценке после {self.max_retries} попыток: {e}", exc_info=True)
                        return {
                            "critical_mistakes": 0,
                            "mistakes": 0,
                            "additional_mistakes": 0,
                            "explanation_critical_mistakes": [],
                            "explanation_mistakes": [],
                            "explanation_additional_mistakes": []
                        }

    def calculate_summary(self, results: List[Dict[str, Any]], dataset_len: int) -> Dict[str, Any]:
        """Вычисляет статистику по результатам"""
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
        
        return {
            "total_critical_mistakes": total_critical_mistakes,
            "total_mistakes": total_mistakes,
            "total_additional_mistakes": total_additional_mistakes,
            "total_all_mistakes": total_all_mistakes,
            "total_tokens": total_tokens,
            "critical_mistakes_per_1000_tokens": round(critical_mistakes_per_1000, 2),
            "mistakes_per_1000_tokens": round(mistakes_per_1000, 2),
            "additional_mistakes_per_1000_tokens": round(additional_mistakes_per_1000, 2),
            "all_mistakes_per_1000_tokens": round(all_mistakes_per_1000, 2),
            "total_dialogs": len(results),
            "successful_dialogs": len(valid_results),
            "failed_dialogs": len(results) - len(valid_results)
        }