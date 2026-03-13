# 1,000 Labels Is All You Need: Sample Efficiency of Fine-Tuned LLMs on Low-Entropy Classification Tasks

**Authors**: [Author Names]
**Affiliation**: [University/Institution]

---

## Abstract

How many labeled examples does a fine-tuned LLM need to master a domain-specific classification task? We investigate sample efficiency on the SALAD cybersecurity dataset by training Qwen3.5-9B with 1K, 5K, 10K, and 20K samples. Our key finding distinguishes two learning processes: *semantic learning* saturates at just 1K samples (normalized F1 = 100% at all sizes), while *label vocabulary learning* follows a non-monotonic curve that requires 20K samples for perfect compliance (strict F1: 1K→77.8%, 5K→55.7%, 10K→86.6%, 20K→100%). Counterintuitively, 5K training produces MORE hallucination than 1K — additional capacity enables pre-training knowledge to override the task schema. These results reveal that scaling labeled data does not improve understanding but rather teaches the model to use the correct vocabulary, fundamentally changing how we should think about training size recommendations for domain-specific LLM deployment.

**Keywords**: sample efficiency, label hallucination, scaling laws, fine-tuning, cybersecurity NLP

---

## 1. Introduction

The conventional wisdom in machine learning suggests that more training data leads to better performance. For LLMs, this translates to a practical question: how large a labeled dataset must we acquire for a domain-specific task? This question directly impacts deployment cost, annotation budget, and project timeline.

We study this question on SOC alert classification and discover a surprising answer: **understanding requires only 1K samples, but compliance requires 20K.** This distinction arises from what we term the *semantic-compliance gap* — the difference between a model's ability to correctly classify inputs and its ability to express classifications using the exact label vocabulary specified by the task.

Our contributions:
1. **Semantic saturation at 1K**: Fine-tuned LLMs learn the underlying task with minimal data
2. **Non-monotonic hallucination**: 5K produces more label hallucination than 1K
3. **Vocabulary learning requires 20K**: Label compliance scales differently from semantic accuracy
4. **Practical guideline**: Choose training size based on whether strict compliance is required

---

## 2. Related Work

### 2.1 Scaling Laws for LLMs
Kaplan et al. (2020) established neural scaling laws showing performance improves predictably with data, compute, and model size. Hoffmann et al. (2022, Chinchilla) refined compute-optimal ratios. However, these focus on pre-training loss, not downstream task compliance. We show that fine-tuning scaling follows a qualitatively different pattern with non-monotonic behavior.

### 2.2 Sample Efficiency in Fine-Tuning
Zhou et al. (2023, LIMA) demonstrated that 1K high-quality examples suffice for instruction tuning. Our finding echos this for semantic understanding, but reveals that label compliance requires 4× more data. QLoRA (Dettmers et al., 2023) and LoRA (Hu et al., 2022) enable efficient fine-tuning that makes our scaling study practical.

### 2.3 Label Hallucination
Our companion work (P3) identified label vocabulary hallucination — where fine-tuned models predict semantically correct but schema-violating labels. We extend this by showing the relationship between training size and hallucination is non-monotonic, contradicting standard learning curve assumptions.

---

## 3. Experimental Setup

**Model**: Qwen3.5-9B with QLoRA (rank 64, 4-bit quantization)

**Training sizes**: 1K, 5K, 10K, 20K (clean, zero-overlap splits)

**Test**: 9,851 held-out samples (no overlap with any training set)

**Metrics**:
- Strict F1: exact label match, no normalization
- Normalized F1: with documented alias mapping
- Hallucinated label count: unique predicted labels not in true vocabulary

---

## 4. Results

### 4.1 Semantic Saturation

| Train Size | Normalized F1 | Interpretation |
|-----------|:------------:|---------------|
| 1K | 1.000 | ✅ Saturated |
| 5K | 1.000 | ✅ Saturated |
| 10K | 1.000 | ✅ Saturated |
| 20K | 1.000 | ✅ Saturated |

> From a semantic perspective, 1K samples is sufficient. The model understands attack categories, triage logic, and classification at all training sizes.

### 4.2 The Strict Scaling Curve

| Train Size | Strict F1 | Halluc Labels | Primary Hallucination |
|-----------|:---------:|:-------------:|----------------------|
| 1K | 0.778 | 1 | Port Scanning (×~4900) |
| **5K** | **0.557** | **4** | Port Scanning, Backdoors, Bots, L2TP |
| 10K | 0.866 | 1 | Port Scanning (fewer) |
| 20K | **1.000** | **0** | None |

### 4.3 The Non-Monotonic Anomaly

The 5K result (strict F1 = 55.7%) is LOWER than 1K (77.8%). This is counterintuitive but explainable:

- At 1K: model has limited adaptor capacity → conservative, fewer hallucinations
- At 5K: more capacity unlocked → pre-training MITRE knowledge bleeds through more aggressively
- At 10K: model begins learning the correct vocabulary ("Reconnaissance" seen enough times)
- At 20K: complete vocabulary compliance

This mirrors observations in catastrophic forgetting literature, where intermediate training amounts can amplify pre-existing biases before eventually overriding them.

---

## 5. Analysis

### 5.1 What Changes With More Data?

We analyze how the predicted vocabulary evolves:

| Train Size | Unique Pred Labels | Hallucinated | % Hallucinated |
|-----------|:-----------------:|:------------:|:--------------:|
| 1K | 8 | 1 | 12.5% |
| 5K | 12 | 4 | 33.3% |
| 10K | 9 | 1 | 11.1% |
| 20K | 8 | 0 | 0.0% |

### 5.2 Cross-Domain Scaling Prediction

Based on task entropy, we predict scaling requirements for other domains:

| Domain | H(Y) | Classes | Predicted N_sufficient |
|--------|------|---------|----------------------|
| SALAD Atk | 1.24 | 8 | 20K (verified) |
| AG News | 2.00 | 4 | ⏳ |
| GoEmotions | 3.75 | 28 | ⏳ (likely >20K) |
| LedGAR | 6.16 | 100 | ⏳ (likely >>20K) |

---

## 6. Discussion

### 6.1 Practical Recommendations

| Requirement | Training Size | Cost |
|-------------|:------------:|:----:|
| Semantic understanding only | **1K** | ~$0.50 |
| Moderate compliance (>85%) | **10K** | ~$2.24 |
| Perfect compliance (100%) | **20K** | ~$4.48 |
| Compliance via model choice | **5K + reasoning model** | ~$2.24 |

> **Alternative to scaling data**: Use a reasoning model (DeepSeek-R1, Phi4-mini) which achieves strict 100% at just 5K samples without needing 20K.

### 6.2 Threats to Validity

- **Single model family**: Tested on Qwen3.5-9B only. The non-monotonic pattern may not generalize to all architectures.
- **Low entropy task**: H=1.24 bits with 8 classes. Higher-entropy tasks may require different scaling curves.
- **Fixed hyperparameters**: LR=2e-4 and rank=64 held constant. Different hyperparameters could shift the curve.

---

## 7. Conclusion

We demonstrate that sample efficiency in LLM fine-tuning involves two distinct learning processes: semantic learning (saturating at 1K) and vocabulary compliance learning (requiring 20K). The non-monotonic hallucination curve — where medium training sizes produce MORE errors than small ones — challenges the assumption that more data always improves performance. We recommend practitioners choose training size based on their compliance requirements, or alternatively adopt reasoning-oriented models that achieve perfect compliance with minimal data.

---

## References

1. Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized Language Models. *NeurIPS*.
2. Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. *NeurIPS*.
3. Hu, E., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.
4. Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. *arXiv:2001.08361*.
5. Zhou, C., et al. (2023). LIMA: Less Is More for Alignment. *NeurIPS*.

---

## Acknowledgments

Computing resources provided by the NSTDA Supercomputer Center (ThaiSC).
