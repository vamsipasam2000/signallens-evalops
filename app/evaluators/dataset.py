import json
from pathlib import Path

from app.evaluators.types import EvalRecord

DEFAULT_EVAL_SET_PATH = Path(__file__).resolve().parents[1] / "data" / "eval_set.jsonl"


def load_eval_records(path: Path = DEFAULT_EVAL_SET_PATH) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                records.append(EvalRecord(**payload))
            except TypeError as exc:
                raise ValueError(f"Invalid eval record at line {line_number}: {exc}") from exc
    return records

