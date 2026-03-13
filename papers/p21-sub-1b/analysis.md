# P21: Sub-1B Is All You Need? Model Size vs Label Compliance in Domain-Specific Tasks

## Reframed Title
**"Bigger Models Follow Instructions Better: How Model Size Affects Label Hallucination in Fine-Tuned LLMs"**

## New Thesis
Sub-1B models achieve perfect semantic accuracy but fail on strict label compliance. Larger models (≥3.8B) and reasoning models follow label schema significantly better.

## Key Evidence

| Model | Size | Strict F1 | Norm F1 | Follows Schema? |
|-------|------|:---------:|:-------:|:---------------:|
| Qwen3.5-0.8B | 0.8B | 77.8% | 100% | ❌ Hallucinate |
| SmolLM2-1.7B | 1.7B | 77.8% | 100% | ❌ Hallucinate |
| Phi4-mini | 3.8B | **100%** | 100% | ✅ |
| DeepSeek-R1 | 7B | **100%** | 100% | ✅ Reasoning |
| Qwen3-8B | 8B | 60.2% | 99.9% | ❌ Worse! |
| Mistral-7B | 7B | 46.1% | 69.1% | ❌ Worst |

## Key Insight
> Size alone doesn't predict label compliance. **Reasoning capability** (DeepSeek-R1, Phi4) matters more.
> Qwen3-8B is 10× larger than 0.8B but has LOWER strict F1.

## Target: EMNLP Industry Track
