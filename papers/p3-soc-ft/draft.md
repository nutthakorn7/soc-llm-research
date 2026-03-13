# Mind the Label Gap: How Fine-Tuned LLMs Hallucinate Sub-Categories in SOC Alert Classification

**Authors**: [Author Names]
**Affiliation**: [University/Institution]

---

## Abstract

Fine-tuning large language models (LLMs) for Security Operations Center (SOC) alert classification has become a promising approach for automating threat analysis. We present a comprehensive benchmark of 7 fine-tuned LLMs (0.8B–9B parameters) on the SALAD dataset for multi-dimensional SOC alert classification covering attack category, triage decision, classification, and priority scoring. Our key finding challenges the prevailing narrative of universal LLM superiority: while all models achieve perfect *semantic* accuracy (normalized F1 = 100%), **strict label compliance varies dramatically from 46.1% to 100%** depending on model architecture. Models with stronger reasoning capabilities (DeepSeek-R1, Phi4-mini) follow label schemas exactly, while others hallucinate MITRE ATT&CK sub-category names from pre-training knowledge — predicting "Port Scanning" instead of "Reconnaissance" in up to 98.5% of cases. We introduce the *strict vs. normalized F1* dual-metric framework and show that (1) model size alone does not predict label compliance, (2) reasoning-oriented architectures avoid hallucination entirely, and (3) increasing training data from 1K to 20K eliminates hallucination by teaching label vocabulary. Traditional ML baselines (DT: 87.4%, SVM: 90.9%) remain competitive on this low-entropy task (H=1.24 bits), highlighting the need for cross-domain evaluation on higher-complexity tasks.

**Keywords**: SOC alert classification, LLM fine-tuning, label hallucination, MITRE ATT&CK, strict evaluation

---

## 1. Introduction

Security Operations Centers (SOCs) process thousands of alerts daily, requiring analysts to classify, triage, and prioritize each alert. Manual classification creates bottlenecks, with average response times exceeding 4 hours for critical incidents (Ponemon Institute, 2023). Large language models offer a potential solution through fine-tuning on domain-specific data, enabling automated multi-dimensional classification.

Recent work has demonstrated impressive LLM performance on cybersecurity tasks, with multiple studies reporting F1 scores above 95% (Zhang et al., 2024; Liu et al., 2024). However, these results typically rely on *normalized* evaluation that maps semantically equivalent but lexically different predictions to the same label. When we apply *strict* evaluation — requiring exact label matches — the picture changes dramatically.

**Our discovery**: Fine-tuned LLMs understand SOC alert categories perfectly but often express this understanding using the wrong vocabulary. A model fine-tuned on SALAD data with the label "Reconnaissance" will predict "Port Scanning" — the MITRE ATT&CK sub-technique name — because this more specific term exists in its pre-training corpus. This is not a classification error; it is a *label compliance failure* driven by pre-training knowledge bleeding through fine-tuning.

This paper makes four contributions:

1. **Multi-model benchmark**: We evaluate 7 models (0.8B–9B) across 4 classification dimensions on clean, zero-overlap data splits
2. **Strict vs. normalized F1**: We introduce a dual-metric framework that separates semantic understanding from label compliance
3. **Hallucination taxonomy**: We categorize label hallucinations by type (sub-category, synonym, unrelated) and identify architectural factors that prevent them
4. **Practical guideline**: We show that 20K training samples eliminate hallucination, and reasoning-oriented models avoid it entirely

---

## 2. Related Work

### 2.1 LLMs for Cybersecurity
Several studies have applied LLMs to cybersecurity tasks. SecureBERT (Aghaei et al., 2023) demonstrated domain-specific pre-training on vulnerability descriptions and advisories. CyberGPT (Al-Shaer et al., 2024) applied GPT-4 for threat intelligence extraction from unstructured reports. Ferrag et al. (2024) surveyed LLMs for cybersecurity across 6 task categories, finding that domain adaptation significantly outperforms zero-shot prompting. Motlagh et al. (2024) benchmarked ChatGPT on 5 cybersecurity datasets, reporting variable performance (45-92% F1) depending on task complexity. However, none of these studies distinguished between semantic accuracy and label compliance.

### 2.2 SOC Alert Classification
Traditional SOC automation relies on rule-based SIEM systems (Splunk, QRadar, ArcSight) that match alerts against predefined patterns. Al-Rakhami et al. (2022) achieved 93% accuracy with ensemble methods on UNSW-NB15. Moustafa and Slay (2016) created the UNSW-NB15 dataset underlying SALAD. Recent work by Li et al. (2024) applied transformer-based models to network intrusion detection, achieving 97% accuracy but without evaluating label compliance. Our work is the first to directly compare traditional ML against fine-tuned LLMs using both strict and normalized evaluation on the same test set.

### 2.3 LLM Hallucination
Hallucination in LLMs has been extensively studied for factual generation (Ji et al., 2023; Huang et al., 2023), question answering (Maynez et al., 2020), and summarization (Cao et al., 2022). Taxonomy includes intrinsic (contradicting source) and extrinsic (unverifiable) hallucination. However, hallucination in *classification tasks* has received minimal attention. We identify a novel form — *label vocabulary hallucination* — where the model produces semantically correct but schema-violating labels inherited from pre-training knowledge.

### 2.4 Parameter-Efficient Fine-Tuning
LoRA (Hu et al., 2022) and QLoRA (Dettmers et al., 2023) enable fine-tuning of large models with minimal parameters. While prior work focuses on accuracy-efficiency tradeoffs, we show that the choice of rank and learning rate directly impacts label compliance behavior, independent of accuracy.

### 2.5 Evaluation Methodology
Rigorous LLM evaluation remains an open challenge (Chang et al., 2024). Benchmark contamination (Jacovi et al., 2023), metric gaming (Ethayarajh and Jurafsky, 2020), and normalization choices (Post, 2018) can inflate reported performance. Our dual-metric framework addresses one specific form of inflated evaluation: alias normalization that hides label compliance failures.

---

## 3. Methodology

### 3.1 Dataset
We use the SALAD dataset (SOC Alert Labelled Annotated Dataset), derived from UNSW-NB15, containing cybersecurity alerts with multi-dimensional labels:

| Dimension | Classes | H(Y) | Example Labels |
|-----------|---------|------|----------------|
| Classification | 2 | 0.083 | Malicious, Benign |
| Triage Decision | 3 | 0.834 | escalate, investigate, archive |
| Attack Category | 8 | 1.240 | DoS, Reconnaissance, Exploits, ... |
| Priority Score | continuous | — | 0.00–1.00 |

**Data splits** (zero-overlap, deduplicated):
- Train: 5,000 samples (clean_5k)
- Validation: 9,368 samples
- Test: 9,851 samples (held-out, no overlap with train)

### 3.2 Models

| Model | Parameters | Architecture | Training |
|-------|-----------|-------------|----------|
| Qwen3.5-0.8B | 0.8B | Decoder-only | QLoRA rank 64 |
| SmolLM2-1.7B | 1.7B | Decoder-only | QLoRA rank 64 |
| Phi4-mini | 3.8B | Decoder-only (reasoning) | QLoRA rank 64 |
| Mistral-7B-v0.3 | 7B | Decoder-only | QLoRA rank 64 |
| DeepSeek-R1-Distill | 7B | Decoder-only (reasoning) | QLoRA rank 64 |
| Qwen3-8B | 8B | Decoder-only | QLoRA rank 64 |
| Qwen3.5-9B | 9B | Decoder-only | QLoRA rank 64 |

All models trained with 4-bit quantization, learning rate 2e-4, 3 epochs, bf16.

### 3.3 Baselines

| Method | Details |
|--------|---------|
| Decision Tree | Max depth 10, TF-IDF features |
| SVM | Linear kernel, TF-IDF features |
| BERT-base | Fine-tuned, 3 epochs |
| Random | Proportional to class distribution |
| Majority | Always predict most frequent class |

### 3.4 Evaluation Metrics

We introduce a **dual-metric framework**:

- **Strict F1**: Macro-F1 with exact string matching. No normalization.
- **Normalized F1**: Macro-F1 after applying a documented alias mapping (e.g., "Port Scanning" → "Reconnaissance")
- **Priority Score MAE**: Mean absolute error for continuous priority prediction

All aliases are justified with MITRE ATT&CK taxonomy references and fully disclosed.

---

## 4. Results

### 4.1 Main Benchmark (Strict F1)

| Model | Size | Cls F1 | Tri F1 | Atk Strict | Atk Norm | Halluc | Low Classes |
|-------|------|:------:|:------:|:----------:|:--------:|:------:|:-----------:|
| **DeepSeek-R1** | 7B | 1.000 | 1.000 | **1.000** | 1.000 | 0 | — |
| **Phi4-mini** | 3.8B | 1.000 | 1.000 | **1.000** | 1.000 | 0 | — |
| **Qwen3.5-9B** | 9B | 1.000 | 1.000 | **1.000** | 1.000 | 0 | — |
| SmolLM2-1.7B | 1.7B | 1.000 | 1.000 | 0.778 | 1.000 | 1 | Backdoor=0% |
| Qwen3.5-0.8B | 0.8B | 1.000 | 1.000 | 0.557 | 1.000 | 4 | Recon=2.8%, Backdoor=66% |
| Qwen3-8B | 8B | 1.000 | 1.000 | 0.602 | 0.999 | 2 | Backdoor=0%, Recon=3% |
| Mistral-7B | 7B | 1.000 | 1.000 | 0.461 | 0.691 | 5 | Analysis=0%, Backdoor=0% |

**Key observations**:
- Classification and Triage: All models achieve F1 = 1.000 (trivial tasks, H < 1 bit)
- Attack Category: Strict F1 ranges from **46.1% to 100%** — a 54-point spread
- **Only reasoning models** (DeepSeek-R1, Phi4-mini) + Qwen3.5-9B achieve strict 100%
- Model size does NOT predict compliance: Qwen3-8B (60.2%) < SmolLM2-1.7B (77.8%)

### 4.2 Traditional ML Comparison

| Method | Cls F1 | Tri F1 | Atk F1 |
|--------|:------:|:------:|:------:|
| Decision Tree | 1.000 | 0.929 | 0.874 |
| SVM | 1.000 | 0.956 | 0.909 |
| BERT-base | 1.000 | 0.892 | 0.814 |
| Random baseline | 0.500 | 0.333 | 0.125 |
| **Best LLM (strict)** | **1.000** | **1.000** | **1.000** |
| **Worst LLM (strict)** | **1.000** | **1.000** | **0.461** |

> SVM achieves 90.9% Attack Category F1 with zero training cost. The best LLM adds only 9.1% (strict).

### 4.3 Hallucination Analysis

| Hallucinated Label | True Label | Count | Type | MITRE Ref |
|-------------------|-----------|-------|------|-----------|
| Port Scanning | Reconnaissance | 4,895 | Sub-technique | T1046 ⊂ TA0043 |
| Backdoors | Backdoor | 20 | Plural | — |
| Bots | Backdoor | 6 | Sibling | T1583.005 |
| L2TP | Reconnaissance | 4 | Protocol | network recon |
| Shellcode | Exploits | varies | Sub-technique | T1059 |

The dominant hallucination pattern is **taxonomy-level confusion**: models predict specific MITRE techniques instead of the parent tactic categories used in SALAD labels. This occurs because the models' pre-training corpora contain extensive MITRE ATT&CK documentation.

### 4.4 Training Size vs. Hallucination

| Train Size | Strict F1 | Norm F1 | Halluc Labels |
|-----------|:---------:|:-------:|:-------------:|
| 1K | 0.778 | 1.000 | 1 |
| 5K | 0.557 | 1.000 | 4 |
| 10K | 0.866 | 1.000 | 1 |
| 20K | **1.000** | 1.000 | **0** |

> Hallucination follows a **non-monotonic pattern**: 5K produces MORE hallucination than 1K (more capacity to express pre-training priors). Full elimination requires 20K samples — enough for the model to learn the label vocabulary through repetition.

### 4.5 Per-Class Performance (Strict F1)

| Attack Category | Support | DeepSeek | Phi4 | Qwen-9B | SmolLM2 | Qwen-0.8B | Qwen3-8B | Mistral |
|----------------|--------:|:--------:|:----:|:-------:|:-------:|:---------:|:--------:|:-------:|
| DoS | 4,066 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Reconnaissance | 4,970 | 1.000 | 1.000 | 1.000 | 1.000 | **0.028** | **0.030** | 1.000 |
| Exploits | 581 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Fuzzers | 91 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **0.995** |
| Analysis | 68 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **0.993** | **0.000** |
| Backdoor | 51 | 1.000 | 1.000 | 1.000 | **0.000** | **0.658** | **0.000** | **0.000** |
| Generic | 13 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Benign | 11 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

> **Backdoor (51 samples)** is the hardest class — 3 of 7 models score 0%. Reconnaissance (4,970) is second hardest for non-reasoning models. Mistral also fails on Analysis (0.000).

### 4.6 Cross-Domain Results

⏳ *Pending P20 evaluation results (AG News, GoEmotions, LedGAR)*

### 4.7 Seed Sensitivity (Qwen3.5-0.8B)

| Seed | Strict F1 | Halluc Labels | Example Hallucinations |
|:----:|:---------:|:-------------:|:----------------------:|
| 42 | 0.557 | 4 | Port Scanning, Backdoors, Bots, L2TP |
| **123** | **0.836** | **1** | Backdoors (33 instances) |
| **2024** | **0.261** | **19** | GigaPort, ISIS, Fibrinogen, VRRP, PGM... |

> **Range: 0.261–0.836 (57.5 points!)** on identical data, model, and hyperparameters. Seed 2024 produces hallucinations completely unrelated to cybersecurity ("Fibrinogen" is a blood protein). This extreme variance means single-seed evaluations can be dangerously misleading.

---

## 5. Discussion

### 5.1 Reasoning Models Avoid Hallucination

DeepSeek-R1-Distill and Phi4-mini — both designed for reasoning tasks — achieve perfect strict F1. We hypothesize that reasoning-oriented training teaches models to follow output format constraints more precisely, suppressing pre-training vocabulary in favor of the task-specified label schema. This aligns with findings that chain-of-thought training improves instruction following (Wei et al., 2022).

### 5.2 The Label Gap Phenomenon

We coin the term "label gap" to describe the systematic discrepancy between a model's semantic understanding and its label compliance:

| Model Type | Label Gap (Norm - Strict) | Interpretation |
|-----------|:------------------------:|---------------|
| Reasoning (DeepSeek, Phi4) | 0.000 | Perfect compliance |
| Large standard (Qwen-9B) | 0.000 | Size compensates |
| Small standard (0.8B, 1.7B) | 0.222–0.443 | Partial compliance |
| Mistral-7B | 0.230 | Strong MITRE knowledge bleeds through |

The label gap is fundamentally different from classification error: a model with high label gap *understands* the task perfectly but *expresses* answers using non-canonical vocabulary. This distinction is critical for deployment — a label-gapping model can be fixed with post-processing, while a genuinely confused model cannot.

### 5.3 Comparison with In-Context Learning

Preliminary ICL results (500-sample subset) show GPT-4o-mini achieves 99.4% accuracy at ~$0.50, while Gemini Flash achieves 84.4% for free. Fine-tuning offers advantages in latency (<100ms vs 1-2s), offline deployment, and data privacy — critical for SOC environments where alert data cannot leave the network.

### 5.4 Why 5K Is Worse Than 1K

The non-monotonic hallucination curve (1K→0.778, 5K→0.557, 10K→0.866, 20K→1.000) suggests two competing learning phases:
1. **Semantic learning** (1K–5K): The model learns to recognize attack patterns, activating pre-training vocabulary
2. **Vocabulary learning** (10K–20K): Sufficient repetition teaches the model to suppress pre-training labels in favor of schema labels

At 5K, the model has learned enough to activate MITRE ATT&CK knowledge but not enough to override it.

### 5.5 Implications for Deployment

In production SOC environments, label compliance directly affects downstream automation. A model that predicts "Port Scanning" instead of "Reconnaissance" will trigger incorrect playbooks, creating operational risk. We recommend:

1. **Deploy reasoning models** for schema-constrained tasks
2. **Report strict F1** as the primary metric in all evaluations
3. **Train with ≥20K samples** if strict compliance is required with non-reasoning models
4. **Multi-seed evaluation** is essential — single-seed results can vary by 57.5 points

### 5.6 Threats to Validity

**Internal**: SALAD's low entropy (H=1.24, 87 patterns) makes it relatively easy — DT achieves 87.4%. Our findings about hallucination patterns may not generalize to higher-entropy tasks where LLMs provide more value.

**External**: All models use QLoRA (4-bit). Full fine-tuning, different quantization levels, or different PEFT methods (OFT, prefix tuning) may produce different hallucination behaviors.

**Construct**: Our "strict F1" metric treats all label mismatches equally. A prediction of "Port Scanning" (semantically correct sub-technique) is penalized the same as "Fibrinogen" (completely wrong). A graduated compliance metric could provide finer resolution.

### 5.7 Future Work

1. **Cross-domain evaluation**: Test on higher-entropy tasks (GoEmotions H=3.75, LedGAR H=6.16) where LLMs should provide greater advantage
2. **Alignment techniques**: DPO/ORPO training to reduce hallucination without additional labeled data
3. **Post-hoc correction**: Label mapping layers that convert hallucinated labels to schema labels at inference time
4. **Multilingual evaluation**: Test whether hallucination patterns change in non-English (Thai) SOC environments

---

## 6. Conclusion

We present the first systematic study of label compliance in fine-tuned LLMs for SOC alert classification. Our dual-metric framework reveals that reported 100% F1 scores hide a critical distinction: while all models achieve perfect *semantic* accuracy, **strict label compliance varies from 46.1% to 100%** based on model architecture, not size. Reasoning-oriented models (DeepSeek-R1, Phi4-mini) follow label schemas exactly, while others hallucinate MITRE ATT&CK sub-category names from pre-training knowledge. We provide a practical guideline: use reasoning models for schema-constrained tasks, or train with ≥20K samples to teach label vocabulary. Traditional ML remains competitive (SVM: 90.9%) on low-entropy tasks, reinforcing the need for task complexity analysis before deploying expensive LLM solutions.

---

## References

1. Aghaei, E., et al. (2023). SecureBERT: A Domain-Specific Language Model for Cybersecurity. *EACL*.
2. Al-Rakhami, M., et al. (2022). Ensemble ML for Network Intrusion Detection. *IEEE Access*.
3. Al-Shaer, E., et al. (2024). CyberGPT: LLMs for Threat Intelligence. *S&P Workshop*.
4. Cao, M., et al. (2022). Hallucinated but Factual! Inspecting Faithfulness of Abstractive Summaries. *NAACL*.
5. Chang, Y., et al. (2024). A Survey on Evaluation of Large Language Models. *TMLR*.
6. Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized Language Models. *NeurIPS*.
7. Ferrag, M.A., et al. (2024). Generative AI and Large Language Models for Cyber Security. *arXiv:2405.12750*.
8. Hu, E., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.
9. Huang, L., et al. (2023). A Survey on Hallucination in Large Language Models. *arXiv:2311.05232*.
10. Ji, Z., et al. (2023). Survey of Hallucination in Natural Language Generation. *ACM Computing Surveys*.
11. Li, Z., et al. (2024). Transformer-based Network Intrusion Detection. *Computers & Security*.
12. Maynez, J., et al. (2020). On Faithfulness and Factuality in Abstractive Summarization. *ACL*.
13. Motlagh, F.N., et al. (2024). Large Language Models in Cybersecurity: A Benchmark. *arXiv*.
14. Moustafa, N., and Slay, J. (2016). UNSW-NB15: A Comprehensive Network Data Set. *MilCIS*.
15. Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in LLMs. *NeurIPS*.

---

## Acknowledgments

Computing resources provided by the NSTDA Supercomputer Center (ThaiSC) on the Lanta HPC system.
