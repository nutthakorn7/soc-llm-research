# P23: Quantization-Aware Fine-Tuning for Edge Security Analytics

## Thesis
4-bit QLoRA enables sub-1B LLMs on edge hardware (T4, Jetson Orin) with <100ms latency, >95% F1.

## Current Data

| Model | Quant | F1 (SALAD) | Status |
|-------|-------|------------|--------|
| Qwen3.5-0.8B | 4-bit (QLoRA) | **100%** | ✅ baseline |
| Qwen3.5-0.8B | 8-bit (QLoRA) | 🔄 eval submitted | ✅ trained |
| Qwen3.5-0.8B | 2-bit | ❌ BnB unsupported | Need GPTQ |
| Qwen3.5-0.8B | 3-bit | ❌ BnB unsupported | Need GPTQ |

## ⚠️ Issue
BitsAndBytes only supports 4-bit and 8-bit for QLoRA training.
2/3-bit requires post-training quantization (GPTQ/AWQ) — different pipeline.

## Edge Hardware Targets

| Device | VRAM | Max Model | Price |
|--------|------|-----------|-------|
| T4 | 16GB | All | $2,000 |
| Jetson Orin NX | 8GB | 0.8B-1.7B | $400 |
| Jetson Orin Nano | 4GB | 0.8B only | $200 |

## Action Plan
- [x] 4-bit QLoRA (baseline) ✅
- [x] 8-bit QLoRA training ✅
- [/] 8-bit eval (submitted)
- [ ] GPTQ 2/3-bit post-training quantization
- [ ] Edge latency benchmarks
- [ ] Write paper

## Target: IEEE IoT Journal (Q1)
