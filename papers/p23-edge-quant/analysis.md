# P23: Quantization-Aware Fine-Tuning for Edge Security Analytics

## Thesis
4-bit QLoRA enables deployment of sub-1B LLMs on edge hardware (T4, Jetson Orin) with <100ms/sample latency while maintaining >95% F1.

## Current Data

| Model | Quant | VRAM | F1 | Latency* |
|---|---|---|---|---|
| Qwen3.5-0.8B | 4-bit (QLoRA) | ~2GB | 100% | ~50ms |
| SmolLM2-1.7B | 4-bit (QLoRA) | ~3GB | 100% | ~80ms |
| Phi-4-mini-3.8B | 4-bit (QLoRA) | ~5GB | 100% | ~120ms |
| DeepSeek-7B | 4-bit (QLoRA) | ~8GB | 100% | ~200ms |

*Estimated, A100. Edge GPU will be 3-5× slower.

## Edge Hardware Targets

| Device | VRAM | Max Model | Price |
|---|---|---|---|
| NVIDIA T4 | 16GB | All | $2,000 |
| Jetson Orin NX | 8GB | 0.8B-1.7B | $400 |
| Jetson Orin Nano | 4GB | 0.8B only | $200 |

## Experiment Plan

### Phase 1: Quantization Ablation (Lanta)
- 0.8B: 2-bit, 3-bit, 4-bit, 8-bit, FP16
- Measure: F1 drop vs memory savings

### Phase 2: Edge Benchmark (local/cloud T4)
- Latency: tokens/sec on T4
- Memory: peak VRAM
- Throughput: alerts/hour

### Phase 3: Cost Model
- Cloud GPU vs Edge GPU over 1 year
- Breakeven analysis

## Key Questions
1. Does 2-bit still work for 0.8B? (aggressive quantization)
2. 0.8B on Jetson Orin Nano ($200) = viable SOC appliance?
3. Batch size impact on edge throughput?

## Target: IEEE IoT Journal (Q1)
