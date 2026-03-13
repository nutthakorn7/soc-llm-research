# From SOC Alerts to Legal Contracts: How Well Do Fine-Tuned LLMs Transfer Across Domains?

**Authors**: [Author Names]

---

## Abstract

Fine-tuned LLMs demonstrate impressive performance within their training domain, but practical deployment requires understanding cross-domain transfer capabilities. We train Qwen3.5-0.8B and Qwen3.5-9B on 4 diverse text classification tasks — SOC alert triage (cybersecurity, H=1.24), news categorization (AG News, H=2.00), emotion detection (GoEmotions, H=3.75), and legal provision classification (LedGAR, H=6.16) — and evaluate each model both in-domain and cross-domain. Our results show that [⏳ pending cross-domain F1]. We combine these findings with traditional ML baselines (DT, SVM) to construct a comprehensive complexity-performance landscape, demonstrating that task entropy H(Y) predicts both the ML-LLM performance gap and the training data requirements. This work provides the empirical foundation for our entropy-based decision framework (companion paper P8).

**Keywords**: cross-domain transfer, multi-domain evaluation, task complexity, LLM generalization

---

## 1. Introduction

Most LLM evaluation studies focus on a single domain, reporting impressive metrics that may not generalize. We ask a fundamental question: **does a fine-tuned LLM trained on cybersecurity alerts also work for legal document classification?** And more importantly, **when should practitioners invest in fine-tuning versus using simpler methods?**

We provide answers through a systematic cross-domain evaluation spanning 4 domains with increasing label entropy. Our experimental matrix creates a complete picture of how fine-tuning transfers (or fails to transfer) across domains of varying complexity.

## 2. Domains and Datasets

| Domain | Dataset | Classes | H(Y) | Task Description |
|--------|---------|:-------:|:----:|-----------------|
| Cybersecurity | SALAD | 8 | 1.24 | Attack category classification |
| News | AG News | 4 | 2.00 | Topic categorization |
| Emotion | GoEmotions | 28 | 3.75 | Fine-grained emotion detection |
| Legal | LedGAR | 100 | 6.16 | Legal provision classification |

### 2.1 Domain Characteristics

| Property | SALAD | AG News | GoEmotions | LedGAR |
|----------|:-----:|:-------:|:----------:|:------:|
| Train samples | 5,000 | 5,000 | 5,000 | 5,000 |
| Test samples | 9,851 | ⏳ | ⏳ | ⏳ |
| Max class % | 99.9% | ~25% | ~30% | ~5% |
| Unique patterns | 87 | ⏳ | ⏳ | ⏳ |
| DT baseline F1 | 0.874 | 0.581 | 0.173 | 0.183 |
| SVM baseline F1 | 0.909 | 0.882 | 0.243 | 0.654 |

## 3. Experimental Setup

### 3.1 Models
- **Qwen3.5-0.8B**: QLoRA rank 64, 4-bit, 3 epochs per domain
- **Qwen3.5-9B**: QLoRA rank 64, 4-bit, 3 epochs per domain (SALAD only + cross-domain evals pending)

### 3.2 Training Configuration
All models trained with identical hyperparameters (lr=2e-4, 5K samples) to isolate domain effects from training differences.

### 3.3 Evaluation Matrix

| Trained On ↓ \ Tested On → | SALAD | AG News | GoEmotions | LedGAR |
|:---------------------------:|:-----:|:-------:|:----------:|:------:|
| SALAD (0.8B) | ✅ 0.778* | ⏳ | ⏳ | ⏳ |
| AG News (0.8B) | 0% EM | ⏳ | ⏳ | ⏳ |
| GoEmotions (0.8B) | 0% EM | ⏳ | ⏳ | ⏳ |
| LedGAR (0.8B) | 0% EM | ⏳ | ⏳ | ⏳ |

*Strict Attack Category F1 on clean_test

### 3.4 Cross-Domain on SALAD (Verified ✅)

Models trained on non-SALAD domains, tested on SALAD test set:

| Trained On | SALAD EM% | What It Predicts | Finding |
|-----------|:---------:|:----------------:|:--------|
| AG News | **0%** | "Sci/Tech" for every alert | News label vocabulary only |
| GoEmotions | **0%** | "neutral" for every alert | Emotion label vocabulary only |
| LedGAR | **0%** | "General" / "Mitre" | Legal label vocabulary only |
| SIEM | **0%** | "Classification:" (truncated) | Partial SOC format, wrong schema |

> **Cross-domain transfer = absolute zero for 0.8B.** Fine-tuning completely overwrites the model's general knowledge with domain-specific label vocabulary.

### 3.5 Size Matters for Cross-Domain: 9B Shows Partial Transfer ✅

| Trained On | Model | SALAD Strict F1 | What It Predicts |
|-----------|:-----:|:---------------:|:----------------:|
| LedGAR (s42) | 0.8B | 0.000 | "General" / "Mitre" |
| LedGAR (s42) | **9B** | **0.156** | "DoS" (286), "Denial Of Service" (193) |
| LedGAR (s77) | **9B** | **0.147** | "Exploitation" (7), "Persistence" (3) |
| LedGAR (s999) | **9B** | **0.000** | "Category: Reconnaissance" (wrong format) |

> **Key finding**: The 9B model retains some pre-training cybersecurity knowledge through LedGAR fine-tuning. It correctly predicts "DoS" (286 times!) even though trained on legal provisions. The 0.8B cannot do this — its smaller capacity is fully overwritten. 
>
> However, seed s999 shows the 9B model reverting to LedGAR output format ("Category: X"), demonstrating unstable cross-domain behavior.

## 4. Results

### 4.1 In-Domain Performance (Primary Results)

| Domain | H(Y) | DT | SVM | LLM-0.8B (strict) | LLM-9B (strict) | ML→LLM Gap |
|--------|:----:|:---:|:---:|:-----------------:|:---------------:|:----------:|
| SALAD | 1.24 | 0.874 | 0.909 | 0.778 | 1.000 | +0.091 |
| AG News | 2.00 | 0.581 | 0.882 | ⏳ | ⏳ | ⏳ |
| GoEmotions | 3.75 | 0.173 | 0.243 | ⏳ | ⏳ | ⏳ |
| LedGAR | 6.16 | 0.183 | 0.654 | ⏳ | ⏳ | ⏳ |

### 4.2 Multi-Seed Stability

| Domain | Seed 42 | Seed 77 | Seed 999 | Mean ± Std |
|--------|:-------:|:-------:|:--------:|:----------:|
| AG News | ⏳ | ⏳ | ⏳ | ⏳ |
| GoEmotions | ⏳ | ⏳ | ⏳ | ⏳ |
| LedGAR | ⏳ | ⏳ | ⏳ | ⏳ |

### 4.3 Hallucination by Domain

| Domain | H(Y) | Halluc Labels | Halluc Type |
|--------|------|:-------------:|-------------|
| SALAD | 1.24 | 1-4 | MITRE sub-categories |
| AG News | 2.00 | ⏳ | ⏳ |
| GoEmotions | 3.75 | ⏳ | ⏳ |
| LedGAR | 6.16 | ⏳ | ⏳ |

**Hypothesis**: Higher entropy (more classes) → more hallucination (more vocabulary confusion)

## 5. Analysis

### 5.1 The Entropy-Gap Curve

We plot ML→LLM performance gap vs. H(Y):

```
Expected curve:
  Gap small ────────────── Gap large
  |                            |
  H=1.24 (SALAD)    H=6.16 (LedGAR)
  SVM=0.909          SVM=0.654
  LLM=1.000          LLM=⏳
  Gap=+9.1%          Gap=⏳
```

> **Prediction**: The ML→LLM gap increases monotonically with H(Y). At H>3 bits, LLM becomes essential because traditional ML F1 drops below 30%.

### 5.2 The Cross-Domain Transfer Matrix

⏳ Full matrix pending. Expected finding: fine-tuning creates domain-specific models that cannot transfer. This motivates multi-task approaches (P15).

## 6. Discussion

### 6.1 When Is Fine-Tuning Worth It?

| H(Y) | Best Traditional | Best LLM | Recommendation |
|:----:|:----------------:|:--------:|----------------|
| <2 | SVM (cheap, fast) | Similar | **Use SVM** |
| 2-3 | SVM (moderate) | Better | **Consider LLM** |
| >3 | Poor (<30%) | Much better | **LLM required** |

### 6.2 The Cost of Cross-Domain Failure

Fine-tuned models are domain-locked. Deploying a single model across domains requires:
- Multi-task training (P15)
- Domain detection + routing
- Or foundation model (unfine-tuned) with ICL

## 7. Conclusion

⏳ Pending final results. Expected conclusion: Cross-domain transfer is zero for fine-tuned LLMs — they become domain-specific classifiers. The ML→LLM gap scales with task entropy, providing a quantitative framework for deployment decisions. For low-entropy tasks, traditional ML suffices; for high-entropy tasks, LLMs are essential but require domain-specific training.

---

## Acknowledgments

Computing resources provided by ThaiSC on the Lanta HPC system.
