# P3: SOC-FT — Fine-Tuning LLMs for SOC Alert Classification: Strict vs Semantic Accuracy

## Reframed Title
**"Mind the Label Gap: How Fine-Tuned LLMs Hallucinate Sub-Categories in SOC Alert Classification"**

## New Thesis
Fine-tuned LLMs achieve perfect semantic accuracy on SOC alert classification but hallucinate MITRE ATT&CK sub-category names instead of parent categories. We introduce strict vs normalized F1 as complementary metrics and show that reasoning-oriented models (DeepSeek-R1, Phi4) avoid hallucination entirely.

## Key Results (Clean Data, Strict Atk F1)

| Model | Size | Strict F1 | Norm F1 | Halluc |
|-------|------|:---------:|:-------:|:------:|
| DeepSeek-R1 | 7B | **100%** | 100% | 0 |
| Phi4-mini | 3.8B | **100%** | 100% | 0 |
| Qwen3.5-9B | 9B | **100%** | 100% | 0 |
| SmolLM2-1.7B | 1.7B | 77.8% | 100% | 1 |
| Qwen3.5-0.8B | 0.8B | 77.8% | 100% | 1 |
| Qwen3-8B | 8B | 60.2% | 99.9% | 2 |
| Mistral-7B | 7B | 46.1% | 69.1% | 5 |

## New Contributions
1. **Strict vs Normalized F1** as dual-metric framework
2. **Reasoning models don't hallucinate** — DeepSeek-R1, Phi4 follow label schema exactly
3. **Hallucination = MITRE sub-technique bleeding** from pre-training
4. **DT baseline 87.4%** on Atk → LLM adds +12.6% (strict) or +9.1% (norm)

## Action Plan
- [x] Clean data multi-model benchmark ✅
- [x] Strict F1 audit ✅
- [ ] Per-class strict F1 table
- [ ] Cross-domain contrast (P20 results)
- [ ] Write draft

## Target: IEEE Access / Expert Systems with Applications
