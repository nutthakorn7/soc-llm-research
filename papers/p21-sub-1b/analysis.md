# P21: Sub-1B is All You Need — Small LLMs for Domain-Specific Tasks

## Thesis
Sub-1B parameter LLMs achieve equal or superior performance to 7B+ models on domain-specific classification tasks when fine-tuned with QLoRA.

## Key Evidence (from existing data)

| Model | Size | SALAD F1 | Train Cost | Inference |
|---|---|---|---|---|
| Qwen3.5-0.8B | **0.8B** | **100%** | $2.24 | 0.0001$/s |
| SmolLM2-1.7B | 1.7B | 100% | $0.60 | — |
| Phi-4-mini | 3.8B | 100% | $0.60 | — |
| DeepSeek-R1-7B | 7B | 100% | $0.70 | — |
| Qwen3-8B | 8B | 99.97% | $0.94 | — |

### Key Finding
> **0.8B = 7B on all 3 tasks.** No benefit from scaling up.
> But strict match: 0.8B = 87.5% vs DeepSeek = 100% → larger models better at exact labels.

## Cross-Domain (TODO: fill after eval)

| Domain | H(Y) | 0.8B F1 | 9B F1 | Gap |
|---|---|---|---|---|
| SALAD (1.24) | 1.24 | 100% | 100% | 0% |
| AG News (2.00) | 2.00 | ⏳ | ⏳ | ⏳ |
| GoEmotions (3.75) | 3.75 | ⏳ | ⏳ | ⏳ |
| LEDGAR (6.16) | 6.16 | ⏳ | ⏳ | ⏳ |

## Hypothesis
> 0.8B ≈ 9B when H(Y) < 4. Gap widens only for H > 5 (LEDGAR).

## Target: EMNLP Industry Track / ACL Findings
