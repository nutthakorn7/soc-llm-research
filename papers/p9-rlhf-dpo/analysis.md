# P9: RLHF/DPO for SOC Alert Reasoning

## Thesis
Reinforcement learning from automated feedback (DPO/ORPO) improves SOC alert classification quality beyond supervised fine-tuning alone.

## Method
- **NO human annotation** — auto-generated preference pairs
- chosen = correct label, rejected = random wrong label
- 5,000 preference pairs from SALAD

## Training Pipeline
```
SALAD (5K) → SFT model (Qwen3.5-0.8B, 100% F1)
         ↓
Auto-generate DPO pairs (chosen/rejected)
         ↓
DPO training (sigmoid loss) → DPO model
ORPO training (odds ratio) → ORPO model
         ↓
Compare: SFT vs DPO vs ORPO
```

## Experiments Submitted

| Method | Model | Loss | LR | Status |
|---|---|---|---|---|
| SFT (baseline) | Qwen3.5-0.8B | CE | 2e-4 | ✅ 100% F1 |
| **DPO** | Qwen3.5-0.8B | sigmoid | 5e-5 | ⏳ Submitted |
| **ORPO** | Qwen3.5-0.8B | odds_ratio | 5e-5 | ⏳ Submitted |

## Key Questions
1. DPO/ORPO > SFT on strict label matching?
2. Better reasoning in generated responses?
3. More robust to adversarial perturbations?

## Action Plan
- [x] Create auto preference data (5K pairs)
- [x] Submit DPO training
- [x] Submit ORPO training
- [ ] Eval both models
- [ ] Compare SFT vs DPO vs ORPO
- [ ] Write paper

## Target: USENIX Security / NeurIPS
