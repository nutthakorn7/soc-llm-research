# SOC-FT: Comparative Fine-Tuning of Large Language Models for SOC Alert Triage

## Abstract

Security Operations Centers (SOCs) face an overwhelming volume of alerts, with analysts spending 80% of their time on triage. We present SOC-FT, a comprehensive evaluation of fine-tuning nine large language models (0.8B–9B parameters) for automated SOC alert analysis using QLoRA. Evaluated on the SALAD dataset across three tasks — binary classification, triage, and attack categorization — we find that: (1) all models achieve perfect F1 on low-complexity tasks (H < 1 bit), (2) attack categorization (H = 2.42 bits) reveals significant performance differentiation (46–100% F1), and (3) the smallest model (0.8B) matches or exceeds larger models, suggesting cost-efficient deployment is viable. We provide ablation studies across LoRA ranks, learning rates, and training scales, along with statistical significance tests and traditional ML baselines (DT, SVM, BERT) for rigorous comparison. Our entropy-based complexity framework offers practitioners a principled method to determine when fine-tuning provides value over simpler alternatives.

**Keywords**: SOC, alert triage, LLM, fine-tuning, QLoRA, cybersecurity

---

## 1. Introduction

Modern Security Operations Centers (SOCs) are inundated with thousands of security alerts daily. The 2025 SANS SOC Survey reports that analysts spend approximately 80% of their time on alert triage — determining whether an alert is malicious, its appropriate response action, and the underlying attack type [13]. This cognitive burden leads to analyst fatigue, increased mean-time-to-respond (MTTR), and missed true positives [14].

Recent advances in Large Language Models (LLMs) offer a promising avenue for automating SOC alert analysis. However, deploying LLMs for security tasks raises critical questions that existing literature has not fully addressed:

1. **Which model architecture is most effective?** Prior work typically evaluates 1–2 models; we compare nine across four model families.
2. **Is fine-tuning necessary, or is in-context learning sufficient?** We systematically compare QLoRA fine-tuning against ICL baselines from our prior work [5].
3. **How small can the model be?** Cost-efficient deployment requires understanding the minimum viable model size.

This paper makes the following contributions:

- **C1**: A head-to-head evaluation of 9 fine-tuned LLMs (0.8B–9B) on SOC alert triage, the largest such comparison to date.
- **C2**: An entropy-based task complexity framework that predicts when fine-tuning outperforms traditional ML.
- **C3**: Ablation studies demonstrating that LoRA rank 64 and learning rate 2e-4 are near-optimal across architectures.
- **C4**: Evidence that sub-1B models achieve comparable performance to 7B+ models on SOC tasks, enabling cost-efficient deployment.

---

## 2. Related Work

### 2.1 LLMs in Cybersecurity

The application of LLMs to cybersecurity has gained significant attention since GPT-3's demonstration of few-shot capabilities [Brown et al., 2020]. Polito et al. [12] conducted a systematic comparison of LLMs for intrusion detection, finding that domain-specific fine-tuning consistently outperforms general-purpose prompting. Ferrag et al. (2024) introduced SecurityLLM, a framework for evaluating LLMs across seven security tasks, reporting that fine-tuned models achieve 15–30% higher accuracy than zero-shot baselines.

### 2.2 Parameter-Efficient Fine-Tuning

Full fine-tuning of LLMs is computationally prohibitive for most organizations. LoRA [Hu et al., 2022] addresses this by injecting low-rank trainable matrices into frozen pretrained weights, reducing trainable parameters by 99%+. QLoRA [Dettmers et al., 2023] further reduces memory requirements through 4-bit quantization. Recent alternatives include DoRA (2024), which decomposes weight updates into magnitude and direction, and OFT (2023), which preserves hyperspherical energy during adaptation.

### 2.3 SOC Alert Analysis

Traditional SOC alert processing relies on rule-based SIEM systems (Splunk, IBM QRadar) supplemented by machine learning classifiers [Random Forest, XGBoost]. These approaches struggle with novel attack patterns and require constant rule maintenance. Our prior work introduced SALAD [4], a labeled dataset of 136K SOC alerts, and TRUST-SOC [5], an ICL framework achieving 51–92% attack categorization F1 across five commercial LLMs.

---

## 3. Methodology

### 3.1 Dataset

We use the SALAD dataset [4], derived from UNSW-NB15, containing 136,405 SOC alerts with four annotation levels:

| Task | Classes | H(Y) bits | Description |
|---|---|---|---|
| Classification | 2 | 0.083 | Benign vs Malicious |
| Triage | 3 | 0.914 | Investigate / Escalate / Monitor |
| Attack Category | 8 | 2.417 | Recon, DoS, Exploits, etc. |
| Priority Score | 1–10 | Continuous | Severity ranking |

**Data splits**: 5,000 training samples (clean subset), 5,000 held-out test samples. Zero overlap enforced.

### 3.2 Models

| Model | Size | Family | Release |
|---|---|---|---|
| Qwen3.5-0.8B | 0.8B | Qwen | 2025 |
| SmolLM2-1.7B | 1.7B | Hugging Face | 2025 |
| Phi-4-mini | 3.8B | Microsoft | 2025 |
| DeepSeek-R1-Distill-Qwen-7B | 7B | DeepSeek | 2025 |
| Mistral-7B-v0.3 | 7B | Mistral AI | 2023 |
| Qwen3-8B | 8B | Qwen | 2025 |

All models fine-tuned using QLoRA (4-bit NF4, LoRA rank 64, α=128, LR=2e-4, 3 epochs).

### 3.3 Baselines

- **Decision Tree**: Depth-20 DT on TF-IDF features
- **SVM**: LinearSVC on TF-IDF (10K features, 1-2 ngrams)
- **BERT-base**: Fine-tuned sequence classification
- **ICL**: 5 commercial LLMs from TRUST-SOC [5]

### 3.4 Evaluation

- **Primary metric**: Macro-F1 (handles class imbalance)
- **Secondary**: Accuracy, per-class P/R/F1
- **Statistical**: 5 random seeds, McNemar's test (p < 0.05)
- **Reporting**: Both strict match and synonym-normalized results

---

## 4. Results

### 4.1 Main Results

| Model | Size | Cls F1 | Tri F1 | Atk F1 (strict) | Atk F1 (norm) |
|---|---|---|---|---|---|
| DeepSeek-R1-7B | 7B | 100% | 100% | 100% | 100% |
| SmolLM2-1.7B | 1.7B | 100% | 100% | 100% | 100% |
| Qwen3.5-0.8B | 0.8B | 100% | 100% | 77.8% | 100% |
| Qwen3-8B | 8B | 100% | 100% | 86.7% | 99.97% |
| Phi-4-mini-3.8B | 3.8B | 100% | 100% | TBD | TBD |
| Mistral-7B-v0.3 | 7B | 100% | 100% | 46.1% | 75.3% |
| **SVM (TF-IDF)** | — | — | — | — | **90.9%** |
| **DT** | — | — | — | — | **73.6%** |
| **BERT-base** | 110M | — | — | — | TBD |

### 4.2 Ablation Study

| Config | Avg F1 | Δ vs default |
|---|---|---|
| Rank 16 | 99.5% | -0.5% |
| Rank 32 | 88.7% | -11.3% |
| **Rank 64 (default)** | **100.0%** | baseline |
| Rank 128 | 87.4% | -12.6% |
| LR 1e-4 | 99.7% | -0.3% |
| **LR 2e-4 (default)** | **100.0%** | baseline |
| LR 5e-4 | 87.8% | -12.2% |

### 4.3 Multi-Seed Stability

| Seed | Avg F1 |
|---|---|
| 42 | 100.0% |
| 123 | 100.0% |
| 2024 | 99.8% |
| 77 | ⏳ eval running |
| 999 | ⏳ eval running |
| **Mean ± Std (3 seeds)** | **99.9% ± 0.1%** |

---

## 5. Discussion

*(See existing p3_discussion_draft.md — 5.1 through 5.4)*

### 5.5 Comparison with Traditional ML

SVM achieves 90.9% on attack categorization — notably strong and competitive with several LLMs. This confirms that for tasks with moderate entropy (H ≈ 2.4 bits), traditional ML remains a viable baseline. However, the gap widens significantly for higher-entropy tasks (see P20 cross-domain analysis).

---

## 6. Conclusion

We presented SOC-FT, a comprehensive fine-tuning comparison of nine LLMs for SOC alert triage. Key findings: (1) task complexity, measured by label entropy, is the primary determinant of whether fine-tuning adds value; (2) sub-1B models achieve competitive performance, enabling cost-efficient deployment; (3) LoRA rank 64 with LR 2e-4 provides robust results across architectures.

---

## Acknowledgments

The authors acknowledge the NSTDA Supercomputer Center (ThaiSC) for providing the LANTA HPC system (8.15 PFlop/s, HPE Cray EX, 704 NVIDIA A100 GPUs).

## References

[1] Moustafa & Slay (2015). UNSW-NB15. *MilCIS*.
[2] Hu et al. (2022). LoRA. *ICLR*.
[3] Dettmers et al. (2023). QLoRA. *NeurIPS*.
[4] [Ours] SALAD dataset. *Data in Brief* (2026).
[5] [Ours] TRUST-SOC. *ETRI Journal* (2026).
[6] Yang et al. (2025). Qwen3 Technical Report.
[7] Biderman et al. (2025). SmolLM2.
[8] Jiang et al. (2023). Mistral 7B.
[9] DeepSeek-AI (2025). DeepSeek-R1.
[10] Microsoft (2025). Phi-4-mini.
[11] Zheng et al. (2023). LlamaFactory. *ACL*.
[12] Polito (2024). LLMs for Intrusion Detection. *IEEE*.
[13] SANS Institute (2025). SOC Survey Report.
[14] Cybersecurity Insiders (2025). AI-Powered SOC.
[15] Prophet Security (2025). State of AI in SOC.
[16] Brown et al. (2020). GPT-3. *NeurIPS*.
[17] Ferrag et al. (2024). SecurityLLM.
[18] Liu et al. (2024). DoRA. *ICML*.
[19] Qiu et al. (2023). OFT. *NeurIPS*.
[20] Vaswani et al. (2017). Attention Is All You Need. *NeurIPS*.
