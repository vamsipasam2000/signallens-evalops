from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int,
) -> float:
    flags = [item_id in relevant_ids for item_id in retrieved_ids]
    return precision_at_k_from_flags(flags, k)


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int,
) -> float:
    flags = [item_id in relevant_ids for item_id in retrieved_ids]
    return recall_at_k_from_flags(flags, len(set(relevant_ids)), k)


def mean_reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
) -> float:
    flags = [item_id in relevant_ids for item_id in retrieved_ids]
    return mean_reciprocal_rank_from_flags(flags)


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int,
) -> float:
    flags = [item_id in relevant_ids for item_id in retrieved_ids]
    return ndcg_at_k_from_flags(flags, total_relevant=len(set(relevant_ids)), k=k)


def precision_at_k_from_flags(relevance_flags: Sequence[bool], k: int) -> float:
    _validate_k(k)
    if not relevance_flags:
        return 0.0
    return _round(sum(relevance_flags[:k]) / k)


def recall_at_k_from_flags(
    relevance_flags: Sequence[bool],
    total_relevant: int,
    k: int,
) -> float:
    _validate_k(k)
    if total_relevant <= 0:
        return 0.0
    bounded_flags = _cap_relevant_flags(relevance_flags, total_relevant)
    return _round(sum(bounded_flags[:k]) / total_relevant)


def mean_reciprocal_rank_from_flags(relevance_flags: Sequence[bool]) -> float:
    for index, is_relevant in enumerate(relevance_flags, start=1):
        if is_relevant:
            return _round(1 / index)
    return 0.0


def ndcg_at_k_from_flags(
    relevance_flags: Sequence[bool],
    total_relevant: int,
    k: int,
) -> float:
    _validate_k(k)
    if total_relevant <= 0:
        return 0.0

    bounded_flags = _cap_relevant_flags(relevance_flags, total_relevant)
    dcg = _discounted_gain(bounded_flags[:k])
    ideal_relevance = [True] * min(total_relevant, k)
    ideal_dcg = _discounted_gain(ideal_relevance)
    if ideal_dcg == 0:
        return 0.0
    return _round(dcg / ideal_dcg)


def _discounted_gain(relevance_flags: Sequence[bool]) -> float:
    return sum(
        (1.0 if is_relevant else 0.0) / math.log2(index + 1)
        for index, is_relevant in enumerate(relevance_flags, start=1)
    )


def _cap_relevant_flags(relevance_flags: Sequence[bool], total_relevant: int) -> list[bool]:
    seen_relevant = 0
    bounded_flags: list[bool] = []
    for is_relevant in relevance_flags:
        if is_relevant and seen_relevant < total_relevant:
            seen_relevant += 1
            bounded_flags.append(True)
            continue
        bounded_flags.append(False)
    return bounded_flags


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be greater than zero")


def _round(value: float) -> float:
    return round(value, 4)
