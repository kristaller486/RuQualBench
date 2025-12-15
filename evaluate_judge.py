import argparse
import asyncio
import json
import logging
import os
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from tqdm.asyncio import tqdm as atqdm

# Import necessary functions from main.py
# We need judge_answer and potentially some setup
from main import judge_answer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

REFERENCE_JUDGE_MODEL = "gemini-2.5-pro"
SEED = 42
SAMPLE_SIZE = 100
LOGS_LIMIT = 100

def get_reference_samples() -> List[Dict[str, Any]]:
    """
    Collects samples from the first 100 logs where the judge was gemini-2.5-pro.
    Returns 100 random samples from this pool.
    """
    logs_dir = Path("logs")
    if not logs_dir.exists():
        raise FileNotFoundError("Logs directory not found")

    # Find all log files
    log_files = list(logs_dir.glob("benchmark_*.json"))
    
    # Sort by timestamp (filename usually contains timestamp)
    # Format: benchmark_YYYY-MM-DD_HH-MM-SS_run_X_dataset.json
    log_files.sort(key=lambda x: x.name)

    valid_logs = []
    
    logger.info(f"Scanning logs for judge '{REFERENCE_JUDGE_MODEL}'...")
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            config = data.get("config", {})
            judge_model = config.get("judge_model", "")
            
            # Check if judge model matches reference
            # We use 'in' because sometimes model names might have prefixes/suffixes
            # or the user might have specified it slightly differently
            if REFERENCE_JUDGE_MODEL.lower() in judge_model.lower():
                valid_logs.append(data)
                
            if len(valid_logs) >= LOGS_LIMIT:
                break
                
        except Exception as e:
            logger.warning(f"Error reading {log_file}: {e}")
            continue

    logger.info(f"Found {len(valid_logs)} logs matching the reference judge.")
    
    if not valid_logs:
        raise ValueError(f"No logs found with judge model '{REFERENCE_JUDGE_MODEL}'")

    # Collect all successful dialog results
    all_samples = []
    for log_data in valid_logs:
        results = log_data.get("results", [])
        for res in results:
            if res.get("error") is None and res.get("answer") is not None:
                # Store necessary data for re-evaluation and comparison
                sample = {
                    "dialog_id": res.get("dialog_id"),
                    "dialog": res.get("dialog"),
                    "answer": res.get("answer"),
                    "original_judge_model": log_data["config"].get("judge_model"),
                    "original_scores": {
                        "critical_mistakes": res.get("critical_mistakes", 0),
                        "mistakes": res.get("mistakes", 0),
                        "additional_mistakes": res.get("additional_mistakes", 0),
                        "explanation_critical_mistakes": res.get("explanation_critical_mistakes", []),
                        "explanation_mistakes": res.get("explanation_mistakes", []),
                        "explanation_additional_mistakes": res.get("explanation_additional_mistakes", [])
                    },
                    "source_log": log_data["config"].get("timestamp")
                }
                all_samples.append(sample)

    logger.info(f"Collected {len(all_samples)} valid samples from these logs.")
    
    if len(all_samples) < SAMPLE_SIZE:
        logger.warning(f"Available samples ({len(all_samples)}) is less than requested size ({SAMPLE_SIZE}). Using all available.")
        return all_samples

    # Select random samples with fixed seed
    random.seed(SEED)
    selected_samples = random.sample(all_samples, SAMPLE_SIZE)
    
    return selected_samples

async def evaluate_sample(sample: Dict[str, Any], judge_model: str, 
                         api_base: str, api_key: str, semaphore: asyncio.Semaphore,
                         max_retries: int, retry_delay: float, extra_body: dict = None) -> Dict[str, Any]:
    """
    Evaluates a single sample with the new judge.
    """
    try:
        judge_result = await judge_answer(
            sample["dialog"],
            sample["answer"],
            judge_model,
            api_base,
            api_key,
            semaphore,
            max_retries,
            retry_delay,
            extra_body
        )
        
        return {
            "sample_id": sample["dialog_id"], # Using dialog_id as ID, though it might not be unique across datasets
            "original_scores": sample["original_scores"],
            "new_scores": {
                "critical_mistakes": judge_result["critical_mistakes"],
                "mistakes": judge_result["mistakes"],
                "additional_mistakes": judge_result["additional_mistakes"],
                "explanation_critical_mistakes": judge_result["explanation_critical_mistakes"],
                "explanation_mistakes": judge_result["explanation_mistakes"],
                "explanation_additional_mistakes": judge_result["explanation_additional_mistakes"]
            },
            "diff": {
                "critical_mistakes": judge_result["critical_mistakes"] - sample["original_scores"]["critical_mistakes"],
                "mistakes": judge_result["mistakes"] - sample["original_scores"]["mistakes"],
                "additional_mistakes": judge_result["additional_mistakes"] - sample["original_scores"]["additional_mistakes"]
            },
            "error": None
        }
    except Exception as e:
        logger.error(f"Error evaluating sample: {e}")
        return {
            "sample_id": sample["dialog_id"],
            "original_scores": sample["original_scores"],
            "new_scores": None,
            "diff": None,
            "error": str(e)
        }

async def run_evaluation(judge_model: str, extra_body: dict = None):
    # 1. Get reference samples
    logger.info("Step 1: Collecting reference samples...")
    samples = get_reference_samples()
    
    # 2. Setup evaluation
    judge_api_base = os.getenv("JUDGE_MODEL_BASE_URL")
    judge_api_key = os.getenv("JUDGE_MODEL_API_KEY")
    judge_max_workers = int(os.getenv("JUDGE_MODEL_MAX_WORKERS", "10"))
    max_retries = int(os.getenv("MAX_RETRIES", "3"))
    retry_delay = float(os.getenv("RETRY_DELAY", "1.0"))
    
    semaphore = asyncio.Semaphore(judge_max_workers)
    
    logger.info(f"Step 2: Evaluating {len(samples)} samples with judge '{judge_model}'...")
    
    tasks = [
        evaluate_sample(
            sample, judge_model, judge_api_base, judge_api_key,
            semaphore, max_retries, retry_delay, extra_body
        )
        for sample in samples
    ]
    
    results = []
    for coro in atqdm.as_completed(tasks, desc="Evaluating", total=len(tasks)):
        res = await coro
        results.append(res)
        
    # 3. Analyze results
    valid_results = [r for r in results if r["error"] is None]
    failed_count = len(results) - len(valid_results)
    
    if failed_count > 0:
        logger.warning(f"{failed_count} evaluations failed.")
        
    # Calculate aggregate stats
    total_diff_critical = sum(r["diff"]["critical_mistakes"] for r in valid_results)
    total_diff_mistakes = sum(r["diff"]["mistakes"] for r in valid_results)
    total_diff_additional = sum(r["diff"]["additional_mistakes"] for r in valid_results)
    
    # Count how often new judge found MORE or LESS errors
    comparison_stats = {
        "critical": {"more": 0, "less": 0, "same": 0},
        "mistakes": {"more": 0, "less": 0, "same": 0},
        "additional": {"more": 0, "less": 0, "same": 0}
    }
    
    for r in valid_results:
        for key in ["critical", "mistakes", "additional"]:
            full_key = f"{key}_mistakes" if key != "mistakes" else "mistakes" # handle naming inconsistency if any, but keys are consistent here
            if key == "critical": full_key = "critical_mistakes"
            if key == "additional": full_key = "additional_mistakes"
            
            diff = r["diff"][full_key]
            if diff > 0:
                comparison_stats[key]["more"] += 1
            elif diff < 0:
                comparison_stats[key]["less"] += 1
            else:
                comparison_stats[key]["same"] += 1

    summary = {
        "judge_model": judge_model,
        "reference_judge": REFERENCE_JUDGE_MODEL,
        "samples_count": len(samples),
        "successful_evals": len(valid_results),
        "total_diff": {
            "critical_mistakes": total_diff_critical,
            "mistakes": total_diff_mistakes,
            "additional_mistakes": total_diff_additional
        },
        "comparison_stats": comparison_stats,
        "timestamp": datetime.now().isoformat()
    }
    
    # 4. Save results
    output_dir = Path("judge_evals")
    output_dir.mkdir(exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_model_name = judge_model.replace("/", "_").replace(":", "_")
    output_file = output_dir / f"judge_eval_{timestamp_str}_{safe_model_name}.json"
    
    output_data = {
        "summary": summary,
        "detailed_results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Results saved to {output_file}")
    
    # Print summary to console
    print("\n" + "="*60)
    print(f"JUDGE EVALUATION REPORT: {judge_model}")
    print("="*60)
    print(f"Reference Judge: {REFERENCE_JUDGE_MODEL}")
    print(f"Samples: {len(samples)}")
    print("-" * 60)
    print("DIFFERENCES (New Judge - Reference Judge):")
    print(f"Total Critical Mistakes Diff: {total_diff_critical:+d}")
    print(f"Total Mistakes Diff:          {total_diff_mistakes:+d}")
    print(f"Total Additional Mistakes Diff: {total_diff_additional:+d}")
    print("-" * 60)
    print("DETAILED COMPARISON (Count of samples):")
    print(f"{'Type':<15} | {'More Errors':<12} | {'Less Errors':<12} | {'Same':<12}")
    print("-" * 60)
    for key in ["critical", "mistakes", "additional"]:
        stats = comparison_stats[key]
        print(f"{key.capitalize():<15} | {stats['more']:<12} | {stats['less']:<12} | {stats['same']:<12}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a judge model against a reference judge")
    parser.add_argument("judge_model", type=str, help="Name of the judge model to evaluate")
    parser.add_argument("--extra-body", type=str, help="JSON string for extra_body parameter")
    
    args = parser.parse_args()
    
    extra_body = None
    if args.extra_body:
        try:
            extra_body = json.loads(args.extra_body)
        except json.JSONDecodeError:
            logger.error("Invalid JSON for --extra-body")
            return

    asyncio.run(run_evaluation(args.judge_model, extra_body))

if __name__ == "__main__":
    main()