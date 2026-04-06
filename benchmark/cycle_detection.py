from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, List, Sequence

import tiktoken


GENERATION_CYCLE_ERROR = "generation_cycle_detected"


@dataclass(frozen=True)
class BPECyclesConfig:
    encoding_name: str = "o200k_base"
    min_response_tokens: int = 256
    tail_sizes: tuple[int, ...] = (256, 512, 768, 1024)
    compression_ratio_threshold: float = 0.1


@dataclass(frozen=True)
class BPECyclesResult:
    method: str
    encoding_name: str
    response_tokens: int
    evaluated_tail_tokens: List[int]
    tail_ratios: Dict[str, float]
    compression_ratio: float | None
    compression_ratio_threshold: float
    best_tail_tokens: int | None
    is_generation_cycle: bool
    skipped_from_evaluation: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@lru_cache(maxsize=1)
def _get_encoding(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def _merge_most_frequent_pair(token_ids: Sequence[int]) -> float:
    sequence = list(token_ids)
    if len(sequence) < 2:
        return 1.0

    original_length = len(sequence)
    next_token_id = max(sequence) + 1

    while len(sequence) >= 2:
        pair_counts = Counter(zip(sequence, sequence[1:]))
        if not pair_counts:
            break

        most_frequent_pair, pair_frequency = pair_counts.most_common(1)[0]
        if pair_frequency < 2:
            break

        merged_token_id = next_token_id
        next_token_id += 1

        merged_sequence: List[int] = []
        cursor = 0
        has_merged = False

        while cursor < len(sequence):
            is_pair_match = (
                cursor < len(sequence) - 1
                and sequence[cursor] == most_frequent_pair[0]
                and sequence[cursor + 1] == most_frequent_pair[1]
            )
            if is_pair_match:
                merged_sequence.append(merged_token_id)
                cursor += 2
                has_merged = True
                continue

            merged_sequence.append(sequence[cursor])
            cursor += 1

        if not has_merged:
            break

        sequence = merged_sequence

    return len(sequence) / original_length


def detect_bpe_cycles(text: str, config: BPECyclesConfig | None = None) -> BPECyclesResult:
    config = config or BPECyclesConfig()
    if not text:
        return BPECyclesResult(
            method="bpe_cycles",
            encoding_name=config.encoding_name,
            response_tokens=0,
            evaluated_tail_tokens=[],
            tail_ratios={},
            compression_ratio=None,
            compression_ratio_threshold=config.compression_ratio_threshold,
            best_tail_tokens=None,
            is_generation_cycle=False,
            skipped_from_evaluation=False,
        )

    encoding = _get_encoding(config.encoding_name)
    token_ids = encoding.encode(text)
    response_tokens = len(token_ids)

    if response_tokens < config.min_response_tokens:
        return BPECyclesResult(
            method="bpe_cycles",
            encoding_name=config.encoding_name,
            response_tokens=response_tokens,
            evaluated_tail_tokens=[],
            tail_ratios={},
            compression_ratio=None,
            compression_ratio_threshold=config.compression_ratio_threshold,
            best_tail_tokens=None,
            is_generation_cycle=False,
            skipped_from_evaluation=False,
        )

    evaluated_tail_tokens: List[int] = []
    tail_ratios: Dict[str, float] = {}
    best_ratio: float | None = None
    best_tail_tokens: int | None = None

    for tail_size in config.tail_sizes:
        if response_tokens < tail_size:
            continue

        tail_token_ids = token_ids[-tail_size:]
        ratio = _merge_most_frequent_pair(tail_token_ids)
        rounded_ratio = round(ratio, 4)

        evaluated_tail_tokens.append(tail_size)
        tail_ratios[str(tail_size)] = rounded_ratio

        if best_ratio is None or ratio < best_ratio:
            best_ratio = ratio
            best_tail_tokens = tail_size

    if best_ratio is None:
        fallback_tail_size = response_tokens
        fallback_ratio = _merge_most_frequent_pair(token_ids)
        evaluated_tail_tokens.append(fallback_tail_size)
        tail_ratios[str(fallback_tail_size)] = round(fallback_ratio, 4)
        best_ratio = fallback_ratio
        best_tail_tokens = fallback_tail_size

    is_generation_cycle = best_ratio <= config.compression_ratio_threshold

    return BPECyclesResult(
        method="bpe_cycles",
        encoding_name=config.encoding_name,
        response_tokens=response_tokens,
        evaluated_tail_tokens=evaluated_tail_tokens,
        tail_ratios=tail_ratios,
        compression_ratio=round(best_ratio, 4),
        compression_ratio_threshold=config.compression_ratio_threshold,
        best_tail_tokens=best_tail_tokens,
        is_generation_cycle=is_generation_cycle,
        skipped_from_evaluation=is_generation_cycle,
    )
