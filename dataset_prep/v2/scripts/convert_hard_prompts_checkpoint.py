import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Конвертация checkpoint hardest prompts v2 в финальные файлы"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Директория запуска с checkpoint_manifest.json и checkpoint_results.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Куда записать итоговые файлы; по умолчанию используется --run-dir"
    )
    return parser.parse_args()


def load_json_file(file_path: Path) -> Any:
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Пропускаю битую строку checkpoint_results.jsonl на строке %s",
                    line_number,
                )
                continue
            if not isinstance(item, dict):
                raise ValueError(f"Некорректная запись в JSONL на строке {line_number}")
            items.append(item)
    return items


def write_json(file_path: Path, payload: Any) -> None:
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


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


def deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_prompt_id: Dict[str, Dict[str, Any]] = {}
    duplicates = 0
    for item in results:
        prompt_id = item.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError("В checkpoint_results.jsonl найдена запись без корректного prompt_id")
        if prompt_id in by_prompt_id:
            duplicates += 1
        by_prompt_id[prompt_id] = item

    if duplicates > 0:
        logger.warning("Найдено %s дубликатов prompt_id, оставляю последнюю запись", duplicates)

    return list(by_prompt_id.values())


def sort_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def calculate_observed_costs(results: List[Dict[str, Any]]) -> tuple[float, float]:
    generation_cost = 0.0
    judge_cost = 0.0
    for item in results:
        for model_result in item.get("model_results", []):
            generation_cost += float(model_result.get("generation_cost_usd") or 0.0)
            judge_cost += float(model_result.get("judge_cost_usd") or 0.0)
    return generation_cost, judge_cost


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


def build_metadata(
    manifest: Dict[str, Any],
    processed_prompts: int,
    run_dir: Path,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "timestamp": manifest.get("timestamp", run_dir.name),
        "seed": manifest.get("seed"),
        "head": manifest.get("head"),
        "limit": manifest.get("limit"),
        "sample_models": manifest.get("sample_models"),
        "judge_model": manifest.get("judge_model"),
        "models_file": manifest.get("models_file"),
        "dataset_sources": manifest.get("dataset_sources", []),
        "processed_prompts": processed_prompts,
        "generation_workers": manifest.get("generation_workers"),
        "judge_workers": manifest.get("judge_workers"),
        "debug_logs": manifest.get("debug_logs"),
        "generated_from_checkpoint": True,
        "checkpoint_status": manifest.get("status"),
    }

    existing_metadata_path = run_dir / "metadata.json"
    if existing_metadata_path.exists():
        existing_metadata = load_json_file(existing_metadata_path)
        if isinstance(existing_metadata, dict):
            existing_metadata.update(metadata)
            metadata = existing_metadata

    return metadata


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    output_dir = args.output_dir or run_dir

    manifest_path = run_dir / "checkpoint_manifest.json"
    checkpoint_results_path = run_dir / "checkpoint_results.jsonl"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Не найден файл {manifest_path}")
    if not checkpoint_results_path.exists():
        raise FileNotFoundError(f"Не найден файл {checkpoint_results_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json_file(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint_manifest.json должен содержать объект")

    results = deduplicate_results(load_jsonl(checkpoint_results_path))
    sort_results(results)

    metadata = build_metadata(manifest, len(results), run_dir)
    run_summary = build_run_summary(results)

    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "run_summary.json", run_summary)
    write_json(output_dir / "prompt_results.json", results)
    write_csv(output_dir / "hardest_prompts.csv", build_csv_rows(results))

    logger.info("Checkpoint не изменялся и не удалялся")
    logger.info("Сконвертировано %s диалогов", len(results))
    logger.info("Результаты сохранены в %s", output_dir)


if __name__ == "__main__":
    main()
