# SALAD-v2: A Multi-Source Cybersecurity Dataset for Cross-Environment SOC Alert Classification

**Authors**: [Author Names]

---

## Abstract

Single-source SOC datasets limit generalization — models trained on one SIEM environment often fail in another. We introduce SALAD-v2, a multi-source cybersecurity classification dataset merging SALAD (UNSW-NB15 derived, 5K train) with Advanced SIEM logs (5K train) to create a 10K-sample, ~14-class benchmark with doubled entropy (H≈2.0 vs. H=1.24). We evaluate Qwen3.5-0.8B on SALAD-v2 and measure: (1) cross-source transfer (SALAD-trained → SIEM test), (2) whether multi-source training improves label compliance, and (3) the entropy increase's effect on traditional ML baselines. [⏳ Evaluation pending.]

---

## 1. Introduction

SALAD's low entropy (H=1.24) and 87 unique patterns make it insufficient as a standalone LLM benchmark. Real SOC environments ingest alerts from multiple sources with different schemas and vocabularies. SALAD-v2 addresses this by combining two cybersecurity data sources into a unified training pipeline.

## 2. Related Work

### 2.1 Multi-Source Datasets
Dataset merging has improved robustness in NLP (Talmor and Berant, 2019). CyberBench (Gupta et al., 2024) combines multiple security tasks but not data sources. We create the first multi-source cybersecurity *classification* dataset.

### 2.2 Dataset Difficulty
Our companion work (P24) shows most cyber datasets are Grade D (trivially easy). SALAD-v2 targets Grade C-B difficulty by doubling entropy from H=1.24 to H≈2.0.

## 3. Dataset Composition

| Property | SALAD | SIEM | **SALAD-v2** |
|----------|:-----:|:----:|:------------:|
| Train samples | 5,000 | 5,000 | **10,000** |
| Test samples | 9,851 | 1,000 | **10,851** |
| Attack classes | 8 | 6 | **~14** |
| H(Y) | 1.244 | 0.847 | **~2.0** |
| Source | UNSW-NB15 | Enterprise SIEM | **Combined** |

### 3.1 Label Mapping

| SALAD Labels | SIEM Labels | Unified |
|-------------|------------|---------|
| DoS | DDoS | DoS |
| Reconnaissance | Port Scan | Reconnaissance |
| Exploits | — | Exploits |
| — | Malware | Malware (new) |
| — | Phishing | Phishing (new) |

## 4. Results

### 4.1 Multi-Source Training

| Model | Train | Test-SALAD | Test-SIEM | Test-v2 |
|-------|:-----:|:----------:|:---------:|:-------:|
| Qwen3.5-0.8B | SALAD only | 0.778 | ⏳ | — |
| Qwen3.5-0.8B | SIEM only | ⏳ | ⏳ | — |
| Qwen3.5-0.8B | **SALAD-v2** | ⏳ | ⏳ | ⏳ |

### 4.2 Entropy Effect on Baselines

| Method | SALAD (H=1.24) | SIEM (H=0.85) | v2 (H≈2.0) |
|--------|:--------------:|:-------------:|:-----------:|
| DT | 0.874 | ⏳ | ⏳ |
| SVM | 0.909 | ⏳ | ⏳ |
| LLM strict | 0.778 | ⏳ | ⏳ |

## 5. Discussion

### 5.1 Cross-Source Challenges
- Schema differences require label normalization
- Feature distributions vary between SIEM products
- Multi-source training may improve robustness but increase hallucination vocabulary

### 5.2 Toward a Harder Benchmark
SALAD-v2 (H≈2.0) begins to enter the transition zone where LLMs add value over SVM. Combined with P8's entropy framework, this provides a testing ground for the 2-3 bit regime.

## 6. Conclusion

⏳ Pending evaluation results. Expected contribution: first multi-source cybersecurity dataset for LLM fine-tuning evaluation, providing a harder benchmark than SALAD alone.

---

## References

1. Gupta, A., et al. (2024). CyberBench: Multi-Task Security Evaluation for LLMs. *arXiv*.
2. Talmor, A. and Berant, J. (2019). MultiQA: An Empirical Investigation of Generalization and Transfer in Reading Comprehension. *ACL*.
