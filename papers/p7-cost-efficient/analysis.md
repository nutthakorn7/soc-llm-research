# P7: Cost-Efficient LLM Deployment for SOC

## Training Cost (A100 80GB, Lanta HPC)

| Model | Size | GPU-hours | Cost* | Atk F1 (clean) |
|-------|------|-----------|-------|----------------|
| SmolLM2-1.7B | 1.7B | **0.30h** | $0.60 | **100%** |
| Phi-4-mini | 3.8B | 0.30h | $0.60 | ⏳ phi4 eval |
| DeepSeek-R1-7B | 7B | 0.35h | $0.70 | **100%** |
| Mistral-7B | 7B | 0.42h | $0.84 | 91.7% |
| Qwen3-8B | 8B | 0.47h | $0.94 | 99.97% |
| Qwen3.5-0.8B | 0.8B | 1.12h | $2.24 | **100%** |

*$2/GPU-hour (A100 cloud pricing)

## Key Finding (updated with clean data)
> SmolLM2 ($0.60) = **cheapest model** that achieves 100% Atk F1.
> Qwen3.5-0.8B ($2.24) costs more due to visual encoder overhead.

## Methods Comparison

| Method | Train Cost | Atk F1 |
|--------|-----------|--------|
| DT (TF-IDF) | ~$0 | 87.4% |
| SVM (TF-IDF) | ~$0 | 90.9% |
| BERT-base | ~$0.20 | 81.4% |
| **QLoRA 0.8B** | **$2.24** | **100%** |
| **QLoRA 1.7B** | **$0.60** | **100%** |

## Action Plan
- [x] Training cost benchmarks ✅
- [x] Clean data multi-model F1 ✅
- [ ] Latency benchmarks (edge deployment)
- [ ] Write paper
