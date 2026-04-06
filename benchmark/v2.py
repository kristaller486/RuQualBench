import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.base import BenchmarkBase
from benchmark.cycle_detection import GENERATION_CYCLE_ERROR, detect_bpe_cycles
from benchmark.judge_v2 import V2Judge, build_empty_mistakes_count

logger = logging.getLogger(__name__)


class BenchmarkV2(BenchmarkBase):
    def __init__(
        self,
        dataset_path: str,
        model_name: Optional[str] = None,
        judge_model_name: Optional[str] = None,
        extra_body: Optional[dict] = None,
        verbose_name: Optional[str] = None,
        debug_logs: int = 0,
    ):
        super().__init__(dataset_path, model_name, judge_model_name, extra_body, verbose_name)
        self.debug_logs = debug_logs
        self.logs_dir = Path("logs_v2")
        self.v2_judge = V2Judge(self.judge_transport, debug_logs=debug_logs)

    async def generate_answers_batch(self, dataset: List[List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        answer_results = await super().generate_answers_batch(dataset)

        for answer_result in answer_results:
            if answer_result["error"] is not None or answer_result["answer"] is None:
                continue

            cycle_detection_result = detect_bpe_cycles(answer_result["answer"])
            answer_result["generation_metadata"] = {
                "cycle_detection": cycle_detection_result.to_dict()
            }

            if cycle_detection_result.is_generation_cycle:
                answer_result["error"] = GENERATION_CYCLE_ERROR
                logger.warning(
                    "Диалог %s исключен из оценки: найден цикл генерации (ratio=%s, tail=%s)",
                    answer_result["dialog_id"],
                    cycle_detection_result.compression_ratio,
                    cycle_detection_result.best_tail_tokens,
                )

        return answer_results

    async def judge_dialog_answer(self, answer_result: Dict[str, Any], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Оценивает готовый ответ диалога (V2)"""
        if answer_result["error"] is not None or answer_result["answer"] is None:
            return {
                **answer_result,
                "mistakes": [],
                "mistakes_count": build_empty_mistakes_count(),
                "splitted_answer": ""
            }
        
        try:
            judge_result = await self.v2_judge.evaluate(
                answer_result["dialog"],
                answer_result["answer"],
                semaphore
            )
            judge_result_dict = judge_result.to_dict()
            
            result = {
                **answer_result,
                "mistakes": judge_result_dict["mistakes"],
                "mistakes_count": judge_result_dict["mistakes_count"],
                "splitted_answer": judge_result_dict["splitted_answer"]
            }
            
            if self.debug_logs >= 1:
                result["raw_judge_response_text"] = judge_result_dict["raw_judge_response_text"]
            
            if self.debug_logs >= 2:
                result["rendered_user_prompt"] = judge_result_dict["rendered_user_prompt"]
                
            return result
        except Exception as e:
            logger.error(f"Ошибка при оценке диалога v2 {answer_result['dialog_id']}: {e}", exc_info=True)
            return {
                **answer_result,
                "mistakes": [],
                "mistakes_count": build_empty_mistakes_count(),
                "splitted_answer": ""
            }

    def calculate_summary(self, results: List[Dict[str, Any]], dataset_len: int) -> Dict[str, Any]:
        """Вычисляет статистику по результатам (V2)"""
        valid_results = [r for r in results if r["error"] is None]
        cycle_results = [r for r in results if r.get("error") == GENERATION_CYCLE_ERROR]
        non_cycle_failed_results = [
            r for r in results if r["error"] is not None and r.get("error") != GENERATION_CYCLE_ERROR
        ]

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
            "failed_dialogs": dataset_len - len(valid_results),
            "generation_cycles_detected": len(cycle_results),
            "generation_failures_excluding_cycles": len(non_cycle_failed_results),
            "skipped_due_to_generation_cycle": len(cycle_results),
        }
