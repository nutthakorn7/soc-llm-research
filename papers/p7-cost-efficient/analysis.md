# P7: Cost-Efficient LLM Deployment for SOC

## Training Cost (A100 80GB, Lanta HPC)

| Model | Size | GPU-hours | Samples/s | Cost* |
|---|---|---|---|---|
| SmolLM2-1.7B | 1.7B | **0.30h** | 13.9 | $0.60 |
| Phi-4-mini-3.8B | 3.8B | 0.30h | 13.8 | $0.60 |
| DeepSeek-R1-7B | 7B | 0.35h | 12.0 | $0.70 |
| Mistral-7B | 7B | 0.42h | 9.8 | $0.84 |
| Qwen3-8B | 8B | 0.47h | 8.9 | $0.94 |
| Qwen3.5-0.8B | 0.8B | 1.12h | 3.7 | $2.24 |
| OFT-9B | 9B | 1.92h | 2.2 | $3.84 |

*Cost estimated at $2/GPU-hour (A100 cloud pricing)

### Key Finding
> **Qwen3.5-0.8B costs $2.24 to fine-tune** but achieves 100% normalized F1.
> SmolLM2 costs $0.60 — **3.7× cheaper** with same performance.

## Scaling Cost

| Train Size | GPU-hours | Cost | F1 (strict) | F1 (norm) |
|---|---|---|---|---|
| 1K | 0.33h | $0.66 | 87.5% | 100% |
| 5K | 1.86h | $3.72 | 83.6% | 100% |
| 10K | 3.61h | $7.22 | 97.4% | 100% |
| 20K | 3.67h | $7.34 | ⏳ | ⏳ |
| 50K | ⏳ running | | | |

### Cost-Performance Sweet Spot
> **5K samples + $3.72** gets you 100% normalized F1.
> Going to 10K ($7.22) only improves strict match by 14%.

## Methods Comparison

| Method | Train Cost | Inference Cost | SALAD F1 | AG News | GoEmo | LEDGAR |
|---|---|---|---|---|---|---|
| DT (TF-IDF) | ~$0 | ~$0 | 73.6% | 57.9% | 16.9% | 18.0% |
| SVM (TF-IDF) | ~$0 | ~$0 | 90.9% | 88.4% | 23.8% | 65.0% |
| **BERT-base** | **~$0.20** | **$0.001/s** | **81.4%** | **92.0%** | **34.0%** | **61.5%** |
| ICL GPT-4o-mini | $0 | $0.01/sample | ⏳ | — | — | — |
| **QLoRA 0.8B** | **$2.24** | **$0.0001/s** | **100%** | ⏳ | ⏳ | ⏳ |
| QLoRA 7B | $0.70 | $0.001/sample | 100% | — | — | — |

### Recommendation
> **BERT wins on cost** ($0.20 vs $2.24) but loses on F1 (81% vs 100%)
> For SOC: **QLoRA 0.8B** = best F1/cost. For low-budget: **SVM** (free, 90.9%!)
