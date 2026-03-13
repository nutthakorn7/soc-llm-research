# P20: When, How, and How Big — Cross-Domain LLM Fine-Tuning

## ⭐ Most Important Paper (Revealed by Critical Analysis)
SALAD saturates at 1K (H=1.24). Higher-entropy domains will show the real LLM advantage.

## Entropy Values

| Domain | Dataset | K | H(Y) bits | H_max | Normalized |
|--------|---------|---|-----------|-------|------------|
| SOC Alerts | SALAD | 8* | **1.244** | 3.70 | 0.336 |
| News | AG News | 4 | **1.999** | 2.00 | 1.000 |
| Emotion | GoEmotions | 28 | **3.745** | 4.81 | 0.779 |
| Legal | LedGAR | 100 | **6.158** | 6.64 | 0.927 |

*8 categories in test set

## Traditional ML Baselines ✅

| Domain | H(Y) | DT | SVM | LR | BERT |
|--------|------|-----|------|-----|------|
| SALAD | 1.24 | 87.4% | **90.9%** | 72.5% | 81.4% |
| AG News | 2.00 | 57.9% | **88.4%** | 87.4% | **92.0%** |
| GoEmotions | 3.75 | 16.9% | **23.8%** | 12.3% | **34.0%** |
| LedGAR | 6.16 | 18.0% | **65.0%** | 53.5% | **61.5%** |

## LLM Fine-tuning Status

| Domain | 0.8B Train | 9B Train | 0.8B Eval | 9B Eval |
|--------|-----------|----------|-----------|---------|
| SALAD | ✅ 100% | ✅ 100% | ✅ | ✅ |
| AG News | ✅ × 3 seeds | ✅ × 3 seeds | 🔄 submitted | 🔄 submitted |
| GoEmotions | ✅ × 3 seeds | ✅ × 3 seeds | 🔄 submitted | 🔄 submitted |
| LedGAR | ✅ × 3 seeds | ✅ × 3 seeds | 🔄 submitted | 🔄 submitted |

**19 eval jobs submitted** — results will reveal:
- Does 0.8B ≈ 9B on all domains?
- Where does LLM advantage over SVM appear?
- How does scaling behave on high-entropy tasks?

## Key Predictions

| H(Y) | Expected SVM→LLM Gap | Expected 0.8B vs 9B Gap |
|-------|----------------------|-------------------------|
| 1.24 | +9% (confirmed) | 0% (confirmed) |
| 2.00 | +5-10% | < 3% |
| 3.75 | **+30-50%** | 5-15% |
| 6.16 | **+20-30%** | 10-20% |

## Action Plan
- [x] Traditional ML baselines (4 domains) ✅
- [x] BERT baselines (4 domains) ✅
- [x] All 19 LLM trainings (0.8B + 9B × 4 domains × 3 seeds)
- [/] 19 eval jobs submitted — running on Lanta
- [ ] Compile cross-domain comparison table
- [ ] Plot: entropy vs F1 (DT, SVM, BERT, 0.8B, 9B)
- [ ] Write paper

## Target: NeurIPS / ICML
