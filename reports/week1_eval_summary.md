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
| classify_risk | 60 | 0.0034 | 0.0030 | 0.0040 | 0.0070 |
| evaluate_output | 60 | 0.0035 | 0.0020 | 0.0020 | 0.0860 |
| generate_explanation | 60 | 0.0017 | 0.0020 | 0.0020 | 0.0030 |
| normalize_content | 60 | 0.0023 | 0.0020 | 0.0020 | 0.0180 |
| recommend_action | 60 | 0.0011 | 0.0010 | 0.0020 | 0.0020 |
| retrieve_policy_context | 60 | 0.0266 | 0.0220 | 0.0302 | 0.1820 |
| total | 60 | 0.7432 | 0.7160 | 0.8741 | 1.0980 |

## Readout

The current baseline catches the direct rule-matching examples and misses selected
edge cases that require broader semantic coverage. This is intentional for Week 1:
the eval set creates visible room for future model and policy improvements
without adding nondeterministic behavior yet.
