from __future__ import annotations

import math
import re
from collections.abc import Sequence

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimensions.")
    if not left:
        return 0.0

    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def keyword_overlap_score(query: str, text: str) -> float:
    query_terms = set(TOKEN_PATTERN.findall(query.lower()))
    text_terms = set(TOKEN_PATTERN.findall(text.lower()))
    if not query_terms or not text_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)
