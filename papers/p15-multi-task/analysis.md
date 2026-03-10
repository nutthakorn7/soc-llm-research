# P15: Multi-Task vs Single-Task Learning for SOC Alerts

## Experiment Design

Train Qwen3.5-0.8B on:
1. **Multi-task**: Classification + Triage + Attack Category simultaneously
2. **Single-task × 3**: Each task separately

## Data

| Config | Train File | Tasks |
|---|---|---|
| Multi-task | train_5k_clean.json | All 3 tasks in one output |
| Single-cls | single_task_cls.json | Classification only |
| Single-tri | single_task_tri.json | Triage only |
| Single-atk | single_task_atk.json | Attack Category only |

## Expected Results

| Config | Cls F1 | Tri F1 | Atk F1 | Total GPU-h |
|---|---|---|---|---|
| Multi-task | 100% | 100% | 100% | 1.12h |
| Single-cls | 100% | — | — | ~0.4h |
| Single-tri | — | 100% | — | ~0.4h |
| Single-atk | — | — | ⏳ | ~0.4h |
| **3× Single total** | | | | **1.2h** |

## Key Questions

1. Does multi-task learning improve attack categorization through knowledge transfer?
2. Is single-task more efficient for specific deployments?
3. Does multi-task introduce task interference?

## Hypotheses

- Multi-task ≈ single-task for simple tasks (Cls, Tri are trivial)
- Multi-task **may help** Attack Category through shared feature learning
- Single-task is better for **narrow deployment** (lower latency, smaller output)

## Status
- Multi-task: ✅ (P3 data)
- Single-task training: Data files exist on Lanta
- Single-task eval: TODO
