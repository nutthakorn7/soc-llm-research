# P9: RLHF/DPO for SOC Alert Reasoning

## Thesis
DPO/ORPO reduces label hallucination (e.g., "Port Scanning" → "Reconnaissance") and improves strict-match F1 over SFT alone.

## ⚠️ Critical Challenge
SFT baseline already achieves **100% Avg F1** on SALAD → DPO/ORPO must show improvement on:
1. **Strict match rate** (exact label text)
2. **Robustness** to adversarial perturbation
3. **Cross-domain transfer** (train SALAD, test other domains)

## Training Pipeline
```
SALAD (5K) → SFT model (Qwen3.5-0.8B, 100% F1)
         ↓
Auto-generate DPO pairs (chosen=correct, rejected=wrong category)
         ↓
DPO training (sigmoid loss) → DPO model
ORPO training (odds ratio) → ORPO model
         ↓
Compare: SFT vs DPO vs ORPO on strict match + robustness
```

## Experiments

| Method | Model | Status | Eval |
|--------|-------|--------|------|
| SFT (baseline) | Qwen3.5-0.8B | ✅ 100% F1 | ✅ |
| **ORPO** | Qwen3.5-0.8B | ✅ trained | 🔄 eval submitted |
| **DPO** | Qwen3.5-0.8B | 🔄 training (4805067) | — |

## Expected Differentiators
Since F1=100% for SFT, DPO/ORPO must show:
1. **Fewer hallucinated sub-categories** (Mistral-style errors)
2. **Better calibrated priority scores** (MAE improvement)
3. **Stronger on cross-domain** or adversarial test sets

## Action Plan
- [x] Create auto preference data (5K pairs)
- [x] Submit ORPO training ✅
- [/] DPO training (running)
- [/] ORPO eval (submitted)
- [ ] Compare SFT vs DPO vs ORPO (strict + normalized)
- [ ] Adversarial robustness test
- [ ] Write paper
