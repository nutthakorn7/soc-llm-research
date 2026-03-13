# LoRA vs. OFT: Orthogonal Fine-Tuning for Label-Compliant Security Classification

**Authors**: [Author Names]

---

## Abstract

LoRA (Low-Rank Adaptation) is the dominant parameter-efficient fine-tuning method for LLMs, but Orthogonal Fine-Tuning (OFT) offers a theoretically motivated alternative that preserves pre-trained representations through orthogonal rotations. We compare LoRA and OFT on Qwen3.5-9B for SOC alert classification across 3 random seeds, evaluating strict F1, training stability, and label hallucination. [⏳ OFT eval results pending.] Our analysis tests whether OFT's representation-preserving property reduces or increases label hallucination — preserving pre-training knowledge could amplify MITRE ATT&CK vocabulary bleed-through, or alternatively, preserve the model's instruction-following capability.

---

## 1. Introduction

The choice between LoRA and OFT involves a fundamental tradeoff:
- **LoRA** adds low-rank updates: `W' = W + BA` — potentially overrides pre-training representations
- **OFT** applies orthogonal rotation: `W' = RW` — preserves angular relationships in pre-training space

For label compliance, this creates an open question: does preserving pre-training help (better instruction following) or hurt (more MITRE vocabulary bleeding)?

## 2. Related Work

### 2.1 Orthogonal Fine-Tuning
Qiu et al. (2023) introduced OFT for text-to-image models, showing it preserves generation quality better than LoRA. Liu et al. (2024) extended OFT with block-diagonal structure (BOFT). No prior work compares LoRA vs. OFT for label compliance in classification.

### 2.2 Representation Preservation
Aghajanyan et al. (2021) showed that pre-training creates a low intrinsic dimensionality for fine-tuning. LoRA exploits this via low-rank updates. OFT preserves it via orthogonal rotation. We test which representation strategy better maintains label compliance.

## 3. Experimental Setup

| Aspect | LoRA | OFT |
|--------|------|-----|
| Model | Qwen3.5-9B | Qwen3.5-9B |
| Trainable params | 174M | ~100M |
| GPU time | 1.12h | 1.92h |
| Cost | $2.24 | $3.84 |
| Seeds | 42, 77, 999 | 42, 77, 999 |

## 4. Results

### 4.1 Performance Comparison

| Method | Seed 42 | Seed 77 | Seed 999 | Mean ± Std |
|--------|:-------:|:-------:|:--------:|:----------:|
| LoRA strict F1 | 1.000 | ⏳ | ⏳ | ⏳ |
| LoRA norm F1 | 1.000 | ⏳ | ⏳ | ⏳ |
| OFT strict F1 | ⏳ | ⏳ | ⏳ | ⏳ |
| OFT norm F1 | ⏳ | ⏳ | ⏳ | ⏳ |

### 4.2 Stability

| Method | Strict F1 Std | Halluc Std |
|--------|:------------:|:----------:|
| LoRA | ⏳ | ⏳ |
| OFT | ⏳ | ⏳ |

### 4.3 Hallucination Analysis

| Method | Unique Hallucinations | Dominant Type |
|--------|:--------------------:|:-------------:|
| LoRA | ⏳ | ⏳ |
| OFT | ⏳ | ⏳ |

## 5. Analysis

### 5.1 Hypotheses

**H1: OFT preserves instruction following → fewer hallucinations**
- OFT maintains angular distances between token representations
- If instruction following is encoded as angular relationships, OFT preserves it

**H2: OFT preserves MITRE knowledge → more hallucinations**
- MITRE ATT&CK vocabulary is also encoded in pre-training
- OFT may preserve this "harmful" knowledge that LoRA overrides

### 5.2 Efficiency

| Factor | LoRA | OFT | Winner |
|--------|:----:|:---:|:------:|
| Training time | 1.12h | 1.92h | LoRA (1.7× faster) |
| Trainable params | 174M | ~100M | OFT (fewer params) |
| Memory | Lower (4-bit) | Higher | LoRA |
| Cost | $2.24 | $3.84 | LoRA (42% cheaper) |

## 6. Conclusion

⏳ Pending OFT eval results. This study provides the first comparison of LoRA vs. OFT specifically for label compliance in domain-specific classification.

---

## References

1. Aghajanyan, A., et al. (2021). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. *ACL*.
2. Hu, E., et al. (2022). LoRA: Low-Rank Adaptation. *ICLR*.
3. Liu, S., et al. (2024). BOFT: Parameter-Efficient Orthogonal Finetuning. *ICLR*.
4. Qiu, Z., et al. (2023). Controlling Text-to-Image Diffusion by Orthogonal Finetuning. *NeurIPS*.
