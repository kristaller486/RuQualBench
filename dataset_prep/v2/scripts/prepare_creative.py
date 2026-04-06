import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from datasets import Dataset, load_dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_prep.v2.paths import SOURCES_DIR


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DATASET_SPECS = [
    {
        "dataset_name": "kristaller486/writingprompts-ru",
        "prompt_column": "prompt",
    },
    {
        "dataset_name": "kristaller486/wikisource-creative-ru",
        "prompt_column": "generated_prompt",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Подготавливает creative single-turn промпты для RuQualBench v2"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Какой split загружать у обоих датасетов"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SOURCES_DIR / "creative_ru_prompts.json",
        help="Куда сохранить подготовленный JSON"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Сколько итоговых промптов сохранить после перемешивания"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для перемешивания объединенного пула"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Директория кеша Hugging Face datasets"
    )
    return parser.parse_args()


def write_json(file_path: Path, payload: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_single_dataset(dataset_name: str, split: str, cache_dir: Optional[Path]) -> Dataset:
    logger.info("Загружаю датасет %s (%s)", dataset_name, split)
    dataset = load_dataset(
        dataset_name,
        split=split,
        cache_dir=cache_dir.as_posix() if cache_dir else None,
    )
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Ожидался map-style Dataset для {dataset_name}")
    return cast(Dataset, dataset)


def build_records(dataset: Dataset, dataset_name: str, prompt_column: str) -> List[Dict[str, Any]]:
    if prompt_column not in dataset.column_names:
        raise ValueError(
            f"В датасете {dataset_name} отсутствует колонка {prompt_column!r}. "
            f"Доступные колонки: {dataset.column_names}"
        )

    records: List[Dict[str, Any]] = []
    skipped_empty_prompts = 0

    for row_index, row in enumerate(dataset):
        row_dict = cast(Dict[str, Any], row)
        prompt_value = row_dict.get(prompt_column)
        if not isinstance(prompt_value, str):
            prompt_text = "" if prompt_value is None else str(prompt_value)
        else:
            prompt_text = prompt_value

        prompt_text = prompt_text.strip()
        if not prompt_text:
            skipped_empty_prompts += 1
            continue

        dialog = [{"role": "user", "content": prompt_text}]
        record_id = row_dict.get("id")
        if record_id is None:
            record_id = f"row_{row_index}"

        records.append({
            "dataset_name": dataset_name,
            "prompt_column": prompt_column,
            "row_index": row_index,
            "row_id": str(record_id),
            "prompt_text": prompt_text,
            "dialog": dialog,
        })

    logger.info(
        "Подготовлено %s записей из %s, пропущено пустых промптов: %s",
        len(records),
        dataset_name,
        skipped_empty_prompts,
    )
    return records


def main() -> None:
    args = parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit должен быть положительным числом")

    all_records: List[Dict[str, Any]] = []
    source_stats: List[Dict[str, Any]] = []

    for spec in DATASET_SPECS:
        dataset_name = spec["dataset_name"]
        prompt_column = spec["prompt_column"]
        dataset = load_single_dataset(dataset_name, args.split, args.cache_dir)
        records = build_records(dataset, dataset_name, prompt_column)
        all_records.extend(records)
        source_stats.append({
            "dataset_name": dataset_name,
            "prompt_column": prompt_column,
            "loaded_rows": len(dataset),
            "accepted_rows": len(records),
        })

    rng = random.Random(args.seed)
    rng.shuffle(all_records)

    if args.limit is not None:
        all_records = all_records[:args.limit]

    dialogs: List[List[Dict[str, str]]] = []
    items_metadata: List[Dict[str, Any]] = []
    for item_index, record in enumerate(all_records):
        prompt_id = f"creative_{item_index:06d}_{record['dataset_name'].split('/')[-1]}_{record['row_id']}"
        dialogs.append(record["dialog"])
        items_metadata.append({
            "prompt_id": prompt_id,
            "dataset_name": record["dataset_name"],
            "prompt_column": record["prompt_column"],
            "row_index": record["row_index"],
            "row_id": record["row_id"],
            "prompt_text": record["prompt_text"],
            "turn_count_included": 1,
        })

    summary = {
        "dataset_names": [spec["dataset_name"] for spec in DATASET_SPECS],
        "split": args.split,
        "seed": args.seed,
        "target_limit": args.limit,
        "total_rows_after_merge": sum(item["accepted_rows"] for item in source_stats),
        "saved_rows": len(dialogs),
        "single_turn_only": True,
        "source_stats": source_stats,
    }

    payload = {
        "source": "creative_ru_merged",
        "split": args.split,
        "dialogs": dialogs,
        "items_metadata": items_metadata,
        "summary": summary,
    }
    write_json(args.output, payload)

    logger.info("Сохранено %s single-turn диалогов в %s", len(dialogs), args.output)
    logger.info(
        "Сводка: merged=%s saved=%s seed=%s",
        summary["total_rows_after_merge"],
        summary["saved_rows"],
        args.seed,
    )


if __name__ == "__main__":
    main()
