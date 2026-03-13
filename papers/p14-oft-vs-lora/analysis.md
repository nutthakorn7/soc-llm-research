# P14: OFT vs LoRA for Security Domain Adaptation

## Methods

| Method | Trainable Params | GPU-hours | Approach |
|--------|-----------------|-----------|----------|
| **LoRA** (rank 64) | 174M | 1.12h | Low-rank additive |
| **OFT** | ~100M | 1.92h | Orthogonal rotation |
| **Full FT** | All 9B | TBD | Standard |

## Training Done ✅ (Qwen3.5-9B, 3 seeds each)

| Method | Seed 42 | Seed 77 | Seed 999 | Eval |
|--------|---------|---------|----------|------|
| LoRA | ✅ 100% | ✅ | ✅ | ✅ |
| OFT | ✅ | ✅ | ✅ | 🔄 3 evals submitted |
| Full FT | ✅ | — | — | TODO |

## ⚠️ Challenge
If OFT ≈ LoRA ≈ 100% on SALAD, differentiation must come from:
1. Seed variance (OFT more stable?)
2. Cross-domain performance (use P20 data)
3. Training efficiency (LoRA 1.7× faster)

## Action Plan
- [x] LoRA + OFT training (3 seeds) ✅
- [/] OFT evals submitted (3 jobs)
- [ ] Compare variance across seeds
- [ ] Write paper
