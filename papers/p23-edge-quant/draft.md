# Quantize and Deploy: How Bit-Width Affects Label Hallucination in Edge SOC Models

**Authors**: [Author Names]

---

## Abstract

Edge deployment of SOC classification models demands extreme quantization. We compare 4-bit (QLoRA baseline) and 8-bit quantized Qwen3.5-0.8B on strict F1 and label hallucination for SOC alert classification. Our key question: **does quantization change hallucination patterns?** Lower bit-width constrains model expressivity — similar to lower LoRA rank — which our previous work (P22) showed improves label compliance. We test whether this finding extends to quantization levels and provide deployment guidelines for edge hardware (Jetson Orin, T4).

---

## 1. Introduction

SOC edge devices (Jetson Orin Nano at 4GB VRAM, T4 at 16GB) require aggressive quantization. The standard assumption is that quantization degrades accuracy, but our P22 finding — that reduced capacity can improve label compliance — suggests an alternative: quantization may actually help.

## 2. Related Work

### 2.1 Model Quantization
Dettmers et al. (2022) showed 8-bit training preserves quality. GPTQ (Frantar et al., 2023) and AWQ (Lin et al., 2024) provide post-training quantization for LLMs. We compare training-time quantization (QLoRA at 4-bit/8-bit) for its effect on label compliance.

### 2.2 Edge AI for Security
Real-time SOC classification on edge devices is an emerging requirement. Prior work uses lightweight CNNs; we evaluate quantized LLMs for the first time.

## 3. Results

### 3.1 Quantization Comparison

| Quantization | Strict F1 | Norm F1 | Halluc | Model Size | VRAM |
|:----------:|:---------:|:-------:|:------:|:----------:|:----:|
| 4-bit (QLoRA) | 0.778 | 1.000 | 1 | ~0.5GB | 2GB |
| 8-bit (QLoRA) | ⏳ | ⏳ | ⏳ | ~0.9GB | 4GB |
| 16-bit (full) | — | — | — | 1.6GB | 8GB* |

*Requires A10G or better

### 3.2 Hallucination by Bit-Width

| Bit-Width | "Port Scanning" | Other Halluc | Total Off-Schema |
|:---------:|:---------------:|:------------:|:----------------:|
| 4-bit | 4,895 | 26 | 4,921 |
| 8-bit | ⏳ | ⏳ | ⏳ |

## 4. Edge Deployment Guide

### 4.1 Hardware Compatibility

| Device | VRAM | 4-bit | 8-bit | 16-bit | Cost |
|--------|:----:|:-----:|:-----:|:------:|:----:|
| Jetson Orin Nano | 4GB | ✅ | ✅ | ❌ | $200 |
| Jetson Orin NX | 8GB | ✅ | ✅ | ✅ | $400 |
| T4 | 16GB | ✅ | ✅ | ✅ | $2,000 |
| A10G | 24GB | ✅ | ✅ | ✅ | $3,000 |

### 4.2 Latency Estimates

| Device | 4-bit Latency | 8-bit Latency |
|--------|:-------------:|:-------------:|
| Jetson Orin Nano | ~100ms | ~150ms |
| T4 | ~30ms | ~50ms |
| A10G | ~15ms | ~25ms |

## 5. Analysis

### 5.1 Capacity-Compliance Hypothesis

From P22: lower LoRA rank → better compliance. By analogy:
- 4-bit quantization → less model capacity → may constrain hallucination
- 8-bit quantization → more capacity → may allow more pre-training bleed-through

### 5.2 Deployment Recommendation

| Scenario | Recommended | Why |
|----------|:-----------:|-----|
| Edge SOC (Jetson) | 4-bit + Phi4 | Best compliance, fits in 4GB |
| Data center (T4) | 8-bit | More capacity, fast inference |
| Cloud (A100) | 4-bit QLoRA train → 4-bit deploy | Train and deploy at same bit-width |

## 6. Conclusion

⏳ Pending 8-bit evaluation. Expected contribution: demonstrating the relationship between quantization level and label compliance, providing cost-free compliance improvement through aggressive quantization.

---

## References

1. Dettmers, T., et al. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. *NeurIPS*.
2. Frantar, E., et al. (2023). GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers. *ICLR*.
3. Lin, J., et al. (2024). AWQ: Activation-aware Weight Quantization. *MLSys*.
