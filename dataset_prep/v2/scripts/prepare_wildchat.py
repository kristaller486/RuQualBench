import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple, cast

from datasets import Dataset, load_dataset
from tqdm.auto import tqdm


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

NON_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
MULTISPACE_RE = re.compile(r"\s+")
TRANSLATION_RE = re.compile(r"\b(?:перевод\w*|перевед\w*|перевест\w*)\b", flags=re.IGNORECASE)

ROLE_MAPPING = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "system": "system",
}


@dataclass
class PromptFingerprint:
    prompt_id: str
    normalized_prompt: str
    ngrams: frozenset[Tuple[str, ...]]


class PromptDeduper:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.exact_prompts: set[str] = set()
        self.fingerprints: List[PromptFingerprint] = []
        self.ngram_index: DefaultDict[Tuple[str, ...], List[int]] = defaultdict(list)

    def find_duplicate(
        self,
        normalized_prompt: str,
        ngrams: frozenset[Tuple[str, ...]],
    ) -> Optional[Dict[str, Any]]:
        if normalized_prompt in self.exact_prompts:
            return {
                "kind": "exact",
                "matched_prompt_id": None,
                "score": 1.0,
            }

        if not ngrams:
            return None

        overlap_counts: Counter[int] = Counter()
        for ngram in ngrams:
            for fingerprint_index in self.ngram_index.get(ngram, []):
                overlap_counts[fingerprint_index] += 1

        for fingerprint_index, overlap in overlap_counts.most_common():
            fingerprint = self.fingerprints[fingerprint_index]
            min_len = min(len(ngrams), len(fingerprint.ngrams))
            if min_len == 0:
                continue

            containment = overlap / min_len
            if containment >= self.threshold:
                return {
                    "kind": "trigram",
                    "matched_prompt_id": fingerprint.prompt_id,
                    "score": round(containment, 4),
                }

        return None

    def add(self, prompt_id: str, normalized_prompt: str, ngrams: frozenset[Tuple[str, ...]]) -> None:
        self.exact_prompts.add(normalized_prompt)
        fingerprint_index = len(self.fingerprints)
        self.fingerprints.append(
            PromptFingerprint(
                prompt_id=prompt_id,
                normalized_prompt=normalized_prompt,
                ngrams=ngrams,
            )
        )
        for ngram in ngrams:
            self.ngram_index[ngram].append(fingerprint_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Подготавливает русскоязычные multi-turn промпты из WildChat для RuQualBench v2"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="allenai/WildChat-4.8M",
        help="Название датасета на Hugging Face"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Сплит датасета"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SOURCES_DIR / "wildchat_ru_latest_10000.json",
        help="Куда сохранить подготовленный JSON"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Сколько сэмплов собрать"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Russian",
        help="Какой язык искать у последнего user-turn"
    )
    parser.add_argument(
        "--trigram-threshold",
        type=float,
        default=0.8,
        help="Порог containment для дедупликации по триграммам"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        help="Если задано, отбрасывать диалоги длиннее этого числа turn-ов"
    )
    parser.add_argument(
        "--exclude-translation",
        action="store_true",
        help="Исключать диалоги, где в тексте встречаются паттерны про перевод"
    )
    parser.add_argument(
        "--source-order",
        choices=["sorted-asc", "sort"],
        default="sorted-asc",
        help="`sorted-asc`: считать WildChat отсортированным по timestamp по возрастанию и идти с конца; `sort`: явно сортировать по timestamp"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Директория кеша Hugging Face datasets"
    )
    parser.add_argument(
        "--max-rows-to-scan",
        type=int,
        help="Ограничение на число просмотренных строк для отладки"
    )
    parser.add_argument(
        "--num-proc",
        "--workers",
        dest="num_proc",
        type=int,
        default=8,
        help="Количество процессов для `datasets.map`"
    )
    return parser.parse_args()


def normalize_prompt_text(text: str) -> str:
    lowered = text.lower().replace("\r\n", "\n").replace("\r", "\n")
    without_punctuation = NON_WORD_RE.sub(" ", lowered)
    return MULTISPACE_RE.sub(" ", without_punctuation).strip()


def normalize_role(role: Any) -> Optional[str]:
    if not isinstance(role, str):
        return None
    return ROLE_MAPPING.get(role.strip().lower())


def normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


def is_target_language(value: Any, target_language: str) -> bool:
    if not isinstance(value, str):
        return False

    normalized_value = value.strip().lower()
    normalized_target = target_language.strip().lower()
    if normalized_value == normalized_target:
        return True

    language_aliases = {
        "russian": {"russian", "ru", "ru-ru", "русский"},
    }
    aliases = language_aliases.get(normalized_target)
    if aliases is None:
        return normalized_value == normalized_target
    return normalized_value in aliases


def find_last_user_turn(conversation: Sequence[Dict[str, Any]]) -> Optional[int]:
    for index in range(len(conversation) - 1, -1, -1):
        turn = conversation[index]
        if not isinstance(turn, dict):
            return None
        normalized_role = normalize_role(turn.get("role"))
        if normalized_role == "user":
            return index
    return None


def extract_dialog_slice(conversation: Sequence[Dict[str, Any]], last_user_index: int) -> Optional[List[Dict[str, str]]]:
    dialog: List[Dict[str, str]] = []
    for turn in conversation[:last_user_index + 1]:
        if not isinstance(turn, dict):
            return None
        role = normalize_role(turn.get("role"))
        if role is None:
            return None
        dialog.append({
            "role": role,
            "content": normalize_content(turn.get("content")),
        })

    if not dialog or dialog[-1]["role"] != "user":
        return None
    return dialog


def build_ngrams(normalized_prompt: str, n: int = 3) -> frozenset[Tuple[str, ...]]:
    tokens = normalized_prompt.split()
    if not tokens:
        return frozenset()

    current_n = min(n, len(tokens))
    return frozenset(tuple(tokens[index:index + current_n]) for index in range(len(tokens) - current_n + 1))


def format_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return str(value)


def write_json(file_path: Path, payload: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def build_scan_dataset(dataset: Dataset, source_order: str, max_rows_to_scan: Optional[int]) -> Dataset:
    dataset_len = len(dataset)
    total_to_scan = dataset_len if max_rows_to_scan is None else min(dataset_len, max_rows_to_scan)

    if source_order == "sort":
        logger.info("Явно сортирую датасет по timestamp по убыванию")
        sorted_dataset = dataset.sort("timestamp", reverse=True)
        return cast(Dataset, sorted_dataset.select(range(total_to_scan)))

    if total_to_scan == dataset_len:
        scan_indices = range(dataset_len - 1, -1, -1)
    else:
        scan_indices = range(dataset_len - 1, dataset_len - total_to_scan - 1, -1)

    logger.info("Использую newest-first проход по уже отсортированному датасету")
    return cast(Dataset, dataset.select(scan_indices))


def dialog_text_matches_translation_filter(dialog: Sequence[Dict[str, str]]) -> bool:
    dialog_text = "\n".join(turn.get("content", "") for turn in dialog)
    return bool(TRANSLATION_RE.search(dialog_text))


def preprocess_example(
    example: Dict[str, Any],
    target_language: str,
    max_turns: Optional[int],
    exclude_translation: bool,
) -> Dict[str, Any]:
    result = {
        "status": "skip_no_conversation",
        "conversation_hash": example.get("conversation_hash"),
        "timestamp": format_timestamp(example.get("timestamp")),
        "model": example.get("model"),
        "source_language": example.get("language"),
        "last_user_language": None,
        "last_user_turn_index": -1,
        "last_user_prompt": "",
        "normalized_last_user_prompt": "",
        "dialog_json": "",
        "original_turn_count": 0,
    }

    conversation = example.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        return result

    result["original_turn_count"] = len(conversation)
    last_user_index = find_last_user_turn(conversation)
    if last_user_index is None:
        result["status"] = "skip_no_last_user"
        return result

    dialog = extract_dialog_slice(conversation, last_user_index)
    if dialog is None:
        result["status"] = "skip_invalid_dialog"
        return result

    if max_turns is not None and len(dialog) > max_turns:
        result["status"] = "skip_max_turns"
        return result

    if exclude_translation and dialog_text_matches_translation_filter(dialog):
        result["status"] = "skip_translation"
        return result

    last_user_turn = conversation[last_user_index]
    if not isinstance(last_user_turn, dict):
        result["status"] = "skip_invalid_dialog"
        return result

    prompt_text = dialog[-1]["content"].strip()
    if not prompt_text:
        result["status"] = "skip_empty_last_user_prompt"
        return result

    last_user_language = last_user_turn.get("language")
    result["last_user_language"] = last_user_language
    result["last_user_turn_index"] = last_user_index
    result["last_user_prompt"] = prompt_text

    language_ok = is_target_language(last_user_language, target_language)
    if not language_ok and last_user_language is None:
        language_ok = is_target_language(result["source_language"], target_language)
    if not language_ok:
        result["status"] = "skip_language"
        return result

    normalized_prompt = normalize_prompt_text(prompt_text)
    if not normalized_prompt:
        result["status"] = "skip_empty_last_user_prompt"
        return result

    result["status"] = "candidate"
    result["normalized_last_user_prompt"] = normalized_prompt
    result["dialog_json"] = json.dumps(dialog, ensure_ascii=False)
    return result


def preprocess_dataset(
    scan_dataset: Dataset,
    target_language: str,
    num_proc: int,
    max_turns: Optional[int],
    exclude_translation: bool,
) -> Dataset:
    actual_num_proc = max(1, min(num_proc, len(scan_dataset))) if len(scan_dataset) > 0 else 1
    logger.info("Запускаю `datasets.map` на %s строках с num_proc=%s", len(scan_dataset), actual_num_proc)
    return cast(
        Dataset,
        scan_dataset.map(
            preprocess_example,
            fn_kwargs={
                "target_language": target_language,
                "max_turns": max_turns,
                "exclude_translation": exclude_translation,
            },
            batched=False,
            num_proc=actual_num_proc,
            remove_columns=scan_dataset.column_names,
            load_from_cache_file=False,
            keep_in_memory=False,
            desc="Преобработка WildChat",
        )
    )


def update_summary_for_status(summary: Dict[str, Any], status: str) -> None:
    summary["scanned_rows"] += 1
    if status != "skip_no_conversation":
        summary["rows_with_conversation"] += 1

    if status == "skip_no_last_user":
        summary["rows_without_last_user"] += 1
    elif status == "skip_invalid_dialog":
        summary["rows_with_invalid_dialog"] += 1
    elif status == "skip_max_turns":
        summary["rows_filtered_by_max_turns"] += 1
    elif status == "skip_translation":
        summary["rows_filtered_by_translation"] += 1
    elif status == "skip_empty_last_user_prompt":
        summary["rows_with_empty_last_user_prompt"] += 1
    elif status == "skip_language":
        summary["rows_filtered_by_language"] += 1


def accept_candidate(
    row: Dict[str, Any],
    scan_position: int,
    summary: Dict[str, Any],
    deduper: PromptDeduper,
    dialogs: List[List[Dict[str, str]]],
    items_metadata: List[Dict[str, Any]],
    accepted_limit: int,
) -> bool:
    normalized_prompt = row["normalized_last_user_prompt"]
    ngrams = build_ngrams(normalized_prompt)
    duplicate = deduper.find_duplicate(normalized_prompt, ngrams)
    if duplicate is not None:
        if duplicate["kind"] == "exact":
            summary["rows_deduped_exact"] += 1
        else:
            summary["rows_deduped_trigram"] += 1
        return False

    prompt_id = f"wildchat_{summary['accepted_rows']:05d}_{str(row['conversation_hash'] or scan_position)[:12]}"
    deduper.add(prompt_id, normalized_prompt, ngrams)
    dialog = json.loads(row["dialog_json"])
    dialogs.append(dialog)
    items_metadata.append({
        "prompt_id": prompt_id,
        "scan_position": scan_position,
        "conversation_hash": row["conversation_hash"],
        "timestamp": format_timestamp(row["timestamp"]),
        "model": row["model"],
        "source_language": row["source_language"],
        "last_user_language": row["last_user_language"],
        "last_user_turn_index": row["last_user_turn_index"],
        "last_user_prompt": row["last_user_prompt"],
        "normalized_last_user_prompt": normalized_prompt,
        "turn_count_included": len(dialog),
        "original_turn_count": row["original_turn_count"],
    })
    summary["accepted_rows"] += 1
    return summary["accepted_rows"] >= accepted_limit


def main() -> None:
    args = parse_args()

    if args.limit <= 0:
        raise ValueError("--limit должен быть положительным числом")
    if not 0.0 <= args.trigram_threshold <= 1.0:
        raise ValueError("--trigram-threshold должен быть в диапазоне [0, 1]")
    if args.max_turns is not None and args.max_turns <= 0:
        raise ValueError("--max-turns должен быть положительным числом")
    if args.max_rows_to_scan is not None and args.max_rows_to_scan <= 0:
        raise ValueError("--max-rows-to-scan должен быть положительным числом")
    if args.num_proc <= 0:
        raise ValueError("--num-proc должен быть положительным числом")

    logger.info("Загружаю датасет %s (%s)", args.dataset_name, args.split)
    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
        cache_dir=args.cache_dir.as_posix() if args.cache_dir else None,
    )
    if not isinstance(dataset, Dataset):
        raise TypeError("Ожидался map-style Dataset после load_dataset(..., split=...)")
    dataset = cast(Dataset, dataset)

    scan_dataset = build_scan_dataset(dataset, args.source_order, args.max_rows_to_scan)
    logger.info(
        "Старт подготовки: total_to_scan=%s, num_proc=%s, max_turns=%s, exclude_translation=%s",
        len(scan_dataset),
        args.num_proc,
        args.max_turns,
        args.exclude_translation,
    )
    processed_dataset = preprocess_dataset(
        scan_dataset,
        args.language,
        args.num_proc,
        args.max_turns,
        args.exclude_translation,
    )

    deduper = PromptDeduper(threshold=args.trigram_threshold)
    dialogs: List[List[Dict[str, str]]] = []
    items_metadata: List[Dict[str, Any]] = []

    summary = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "target_language": args.language,
        "max_turns": args.max_turns,
        "exclude_translation": args.exclude_translation,
        "source_order": args.source_order,
        "target_limit": args.limit,
        "num_proc": args.num_proc,
        "trigram_threshold": args.trigram_threshold,
        "scanned_rows": 0,
        "rows_with_conversation": 0,
        "rows_without_last_user": 0,
        "rows_with_invalid_dialog": 0,
        "rows_filtered_by_max_turns": 0,
        "rows_filtered_by_translation": 0,
        "rows_with_empty_last_user_prompt": 0,
        "rows_filtered_by_language": 0,
        "rows_deduped_exact": 0,
        "rows_deduped_trigram": 0,
        "accepted_rows": 0,
    }

    progress = tqdm(total=len(processed_dataset), desc="Отбор WildChat")
    stop_processing = False
    for scan_position, row in enumerate(processed_dataset):
        row_dict = cast(Dict[str, Any], row)
        update_summary_for_status(summary, row_dict["status"])
        if row_dict["status"] == "candidate":
            stop_processing = accept_candidate(
                row=row_dict,
                scan_position=scan_position,
                summary=summary,
                deduper=deduper,
                dialogs=dialogs,
                items_metadata=items_metadata,
                accepted_limit=args.limit,
            )

        progress.update(1)
        progress.set_postfix({"accepted": summary["accepted_rows"]})
        if stop_processing:
            break
    progress.close()

    payload = {
        "source": args.dataset_name,
        "split": args.split,
        "dialogs": dialogs,
        "items_metadata": items_metadata,
        "summary": summary,
    }
    write_json(args.output, payload)

    logger.info("Сохранено %s диалогов в %s", len(dialogs), args.output)
    logger.info(
        "Сводка: scanned=%s accepted=%s max_turns_filtered=%s translation_filtered=%s language_filtered=%s exact_dedup=%s trigram_dedup=%s",
        summary["scanned_rows"],
        summary["accepted_rows"],
        summary["rows_filtered_by_max_turns"],
        summary["rows_filtered_by_translation"],
        summary["rows_filtered_by_language"],
        summary["rows_deduped_exact"],
        summary["rows_deduped_trigram"],
    )


if __name__ == "__main__":
    main()
