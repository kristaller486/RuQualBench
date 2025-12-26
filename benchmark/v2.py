import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from jinja2 import Environment, FileSystemLoader
from litellm import acompletion

from benchmark.base import BenchmarkBase
from benchmark.utils import extract_json_from_response, split_into_numbered_sentences

logger = logging.getLogger(__name__)

class BenchmarkV2(BenchmarkBase):
    def __init__(self, dataset_name: str, model_name: str = None, judge_model_name: str = None, 
                 extra_body: dict = None, verbose_name: str = None, debug_logs: int = 0):
        super().__init__(dataset_name, model_name, judge_model_name, extra_body, verbose_name)
        self.debug_logs = debug_logs
        self.logs_dir = Path("logs_v2")

    async def judge_dialog_answer(self, answer_result: Dict[str, Any], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Оценивает готовый ответ диалога (V2)"""
        if answer_result["error"] is not None or answer_result["answer"] is None:
            return {
                **answer_result,
                "mistakes": [],
                "mistakes_count": {
                    "1": 0,
                    "2": 0,
                    "3": 0
                },
                "splitted_answer": ""
            }
        
        try:
            judge_result = await self._judge_answer(
                answer_result["dialog"],
                answer_result["answer"],
                semaphore
            )
            
            result = {
                **answer_result,
                "mistakes": judge_result["mistakes"],
                "mistakes_count": judge_result["mistakes_count"],
                "splitted_answer": judge_result["splitted_answer"]
            }
            
            if self.debug_logs >= 1:
                result["raw_judge_response_text"] = judge_result["raw_judge_response_text"]
            
            if self.debug_logs >= 2:
                result["rendered_user_prompt"] = judge_result["rendered_user_prompt"]
                
            return result
        except Exception as e:
            logger.error(f"Ошибка при оценке диалога v2 {answer_result['dialog_id']}: {e}", exc_info=True)
            return {
                **answer_result,
                "mistakes": [],
                "mistakes_count": {
                    "1": 0,
                    "2": 0,
                    "3": 0
                },
                "splitted_answer": ""
            }

    async def _judge_answer(self, dialog: List[Dict[str, str]], answer: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Внутренний метод оценки ответа с помощью модели-judge v2"""
        async with semaphore:
            env = Environment(loader=FileSystemLoader('prompts'))
            system_template = env.get_template('judge_system_v2.jinja')
            user_template = env.get_template('judge_user_v2.jinja')
            system_prompt = system_template.render()
            
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
                        # Если вернулся не список, а объект (например, обернутый), попробуем найти список внутри
                        if isinstance(result_list, dict) and "items" in result_list:
                             result_list = result_list["items"]
                        else:
                            # Если все еще не список, считаем что ошибок нет или формат неверный, но для надежности пустой список
                            logger.warning(f"Judge v2 returned non-list response: {result_list}")
                            result_list = []

                    mistakes_count = {
                        "1": 0,
                        "2": 0,
                        "3": 0
                    }
                    
                    for error in result_list:
                        level = error.get("level")
                        if level in [1, 2, 3]:
                            mistakes_count[str(level)] += 1
                    
                    return {
                        "mistakes": result_list,
                        "mistakes_count": mistakes_count,
                        "splitted_answer": splitted_answer,
                        "raw_judge_response_text": raw_response_content,
                        "rendered_user_prompt": user_content
                    }
                except Exception as e:
                    last_error = e
                    error_msg = f"Ошибка при оценке v2 (попытка {attempt+1}/{self.max_retries}): {e}"
                    if 'result_list' in locals():
                         error_msg += f"\nТип ответа: {type(result_list)}"
                         error_msg += f"\nСодержимое (первые 500 симв.): {str(result_list)[:500]}"
                    
                    logger.warning(error_msg)
                    
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    else:
                        logger.error(f"Ошибка при оценке v2 после {self.max_retries} попыток: {e}", exc_info=True)
                        return {
                            "mistakes": [],
                            "mistakes_count": {
                                "1": 0,
                                "2": 0,
                                "3": 0
                            },
                            "splitted_answer": splitted_answer,
                            "raw_judge_response_text": "",
                            "rendered_user_prompt": user_content
                        }

    def calculate_summary(self, results: List[Dict[str, Any]], dataset_len: int) -> Dict[str, Any]:
        """Вычисляет статистику по результатам (V2)"""
        valid_results = [r for r in results if r["error"] is None]
        
        total_critical_mistakes = sum(r["mistakes_count"]["3"] for r in valid_results)
        total_mistakes = sum(r["mistakes_count"]["2"] for r in valid_results)
        total_additional_mistakes = sum(r["mistakes_count"]["1"] for r in valid_results)
        
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
            "total_dialogs": dataset_len,
            "successful_dialogs": len(valid_results),
            "failed_dialogs": dataset_len - len(valid_results)
        }