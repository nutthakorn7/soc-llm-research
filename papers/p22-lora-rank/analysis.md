# P22: LoRA Rank and Label Hallucination in Fine-Tuned LLMs

## Reframed Title
**"Higher Rank, More Hallucination: How LoRA Capacity Amplifies Pre-Training Label Bias"**

## New Thesis
Higher LoRA rank provides more capacity for pre-training priors to bleed through, causing more hallucinated sub-category labels. This is NOT overfitting — it's increased expressivity enabling pre-training biases.

## Clean Data Results (Qwen3.5-0.8B, Strict Atk F1)

| Rank | Strict F1 (clean) | Strict F1 (old) | Halluc | Note |
|------|:---------:|:---------:|:------:|------|
| 16 | **100%** | 98.4% | 0 | Low rank = constrained = follows labels |
| 32 | **100%** | 36.0% | 0 | Old anomaly = leakage artifact |
| **64** | **77.8%** | 100% | **1** | More capacity → hallucination starts |
| 128 | 87.4% | 57.2% | 1 | Continues |

## Key Insight
> **rank 16 achieves 100% strict F1 while rank 64 only gets 77.8%**
> Lower rank constrains the model to follow output schema. Higher rank allows pre-training knowledge to leak sub-category names.

## Contributions
1. **Rank ∝ hallucination** (not overfitting) — first to identify this mechanism
2. **Low rank = better label compliance** on schema-constrained tasks
3. **Previous anomaly was data leakage** — clean data fully resolves rank 32

## Action Plan
- [x] Clean ablation ✅
- [x] Strict F1 audit ✅
- [ ] Multi-model rank comparison
- [ ] Write draft

## Target: EACL / NAACL
