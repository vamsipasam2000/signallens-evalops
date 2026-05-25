# SignalLens EvalOps Week 1 Evaluation Summary

## Scope

- Dataset: `app/data/eval_set.jsonl`
- Eval records: `60`
- Workflow version: `day3-langgraph-langfuse-deterministic-v1`
- Runner: deterministic local LangGraph workflow
- Model calls: none
- Databases: none
- Docker: not included in Week 1
- AWS: not included in Week 1

## Aggregate Metrics

| Metric | Value |
|---|---:|
| Accuracy | `0.8000` |
| Macro F1 | `0.8111` |
| False positive rate | `0.2000` |
| False negative rate | `0.2000` |
| Action agreement | `0.8000` |

## Confusion Matrix

| Expected | safe | spam | harassment | self_harm_sensitive |
|---|---:|---:|---:|---:|
| safe | 12 | 3 | 0 | 0 |
| spam | 3 | 12 | 0 | 0 |
| harassment | 3 | 0 | 12 | 0 |
| self_harm_sensitive | 3 | 0 | 0 | 12 |

## Latency Summary

| Component | Count | Avg ms | P50 ms | P95 ms | Max ms |
|---|---:|---:|---:|---:|---:|
| classify_risk | 60 | 0.0036 | 0.0030 | 0.0040 | 0.0170 |
| evaluate_output | 60 | 0.0033 | 0.0020 | 0.0020 | 0.0790 |
| generate_explanation | 60 | 0.0017 | 0.0020 | 0.0020 | 0.0020 |
| normalize_content | 60 | 0.0021 | 0.0020 | 0.0020 | 0.0030 |
| recommend_action | 60 | 0.0010 | 0.0010 | 0.0010 | 0.0020 |
| retrieve_policy_context | 60 | 0.0230 | 0.0210 | 0.0311 | 0.0350 |
| total | 60 | 0.7183 | 0.7040 | 0.7913 | 0.8500 |

## Readout

The current baseline catches the direct rule-matching examples and misses selected
edge cases that require broader semantic coverage. This is intentional for Week 1:
the eval set creates visible room for future model and policy improvements
without adding nondeterministic behavior yet.
