# P14: OFT vs LoRA for Security Domain Adaptation

## Methods Compared

| Method | Trainable Params | Memory | Approach |
|---|---|---|---|
| **LoRA** | rank × (d_in + d_out) | Low | Low-rank additive |
| **OFT** | rank × rank | Low | Orthogonal rotation |
| **Full FT** | All params | High | Standard fine-tuning |

## Training Results (Qwen3.5-0.8B, 5K samples)

| Method | GPU-hours | Samples/s | Status |
|---|---|---|---|
| LoRA (rank 64) | 1.12h | 3.7/s | ✅ Done |
| OFT | 1.92h | 2.2/s | ✅ Done |
| Full FT | ⏳ | ⏳ | ⏳ Running |

## Eval Results

| Method | Atk F1 (strict) | Atk F1 (norm) | Training Cost |
|---|---|---|---|
| LoRA | 87.5% | 100.0% | $2.24 |
| OFT | ⏳ eval running | ⏳ | $3.84 |
| Full FT | ⏳ eval running | ⏳ | TBD |

## Key Questions
1. Does OFT preserve semantic knowledge better than LoRA?
2. Is Full FT worth 10× compute over LoRA?
3. Which method is most stable across seeds?

## Expected Findings
- LoRA ≈ OFT for this task (not enough data to differentiate)
- Full FT may overfit on 5K samples
- LoRA wins on cost-efficiency: **1.7× faster** than OFT

## Ablation TODO
- OFT with different ranks
- Compare trainable parameter count
