from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MetadataFilter = Mapping[str, Any]


def metadata_matches(metadata: Mapping[str, Any], filters: MetadataFilter | None) -> bool:
    if not filters:
        return True

    for key, expected in filters.items():
        actual = metadata.get(key)
        if isinstance(expected, Mapping):
            if not _operator_match(actual, expected):
                return False
            continue
        if isinstance(expected, list | tuple | set):
            if actual not in expected:
                return False
            continue
        if actual != expected:
            return False

    return True


def metadata_match_count(metadata: Mapping[str, Any], filters: MetadataFilter | None) -> int:
    if not filters:
        return 0
    return sum(
        1
        for key, expected in filters.items()
        if metadata_matches({key: metadata.get(key)}, {key: expected})
    )


def _operator_match(actual: Any, condition: Mapping[str, Any]) -> bool:
    for operator, expected in condition.items():
        if operator == "$eq" and actual != expected:
            return False
        if operator == "$ne" and actual == expected:
            return False
        if operator == "$in" and actual not in expected:
            return False
        if operator == "$contains" and not _contains(actual, expected):
            return False
        if operator not in {"$eq", "$ne", "$in", "$contains"}:
            return False
    return True


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(actual, str):
        return str(expected) in actual
    if isinstance(actual, list | tuple | set):
        return expected in actual
    return False
