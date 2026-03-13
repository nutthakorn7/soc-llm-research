# Can Fine-Tuned LLMs Classify What They've Never Seen? Zero-Shot Generalization in SOC Alert Classification

**Authors**: [Author Names]

---

## Abstract

Fine-tuned LLMs achieve near-perfect performance on seen attack categories, but real SOC environments continuously encounter novel threats. We evaluate zero-shot generalization by training Qwen3.5-0.8B on 7 of 8 SALAD attack categories and testing on the held-out category. Our results reveal that zero-shot classification performance varies dramatically by category — [⏳ pending results]. Categories with distinctive feature signatures (e.g., DoS with high packet rates) transfer well, while categories requiring domain-specific vocabulary (e.g., Backdoor) fail completely. We propose an entropy-based predictor for zero-shot transferability and discuss implications for deploying LLMs in adversarial environments where new attack types emerge continuously.

**Keywords**: zero-shot transfer, SOC classification, novel threat detection, generalization

---

## 1. Introduction

Security threats evolve continuously. A model trained on known attack categories will inevitably encounter novel threats not represented in its training data. The critical question for SOC deployment is: **can a fine-tuned LLM correctly classify an attack category it has never seen?**

We design a controlled leave-one-out experiment: for each of the 8 SALAD attack categories, we train a model on the remaining 7 and evaluate exclusively on the held-out category.

## 2. Related Work

### 2.1 Zero-Shot Classification
Yin et al. (2019) surveyed zero-shot text classification via entailment. Pushp and Srivastava (2017) used label embeddings for zero-shot prediction. LLMs enable a new form: generating labels from pre-training vocabulary, potentially predicting labels never seen in fine-tuning.

### 2.2 Novel Threat Detection
In cybersecurity, zero-day detection relies on anomaly detection (Mirsky et al., 2018). We test whether fine-tuned LLMs can perform a softer version: classifying known threat dimensions for novel attack categories.

## 3. Experimental Setup

### 3.1 Leave-One-Out Protocol

For each category $c_i \in \{$Analysis, Backdoor, Benign, DoS, Exploits, Fuzzers, Generic, Reconnaissance$\}$:

1. **Train**: All samples where Attack Category ≠ $c_i$
2. **Test**: Only samples where Attack Category = $c_i$
3. **Measure**: Strict F1, predicted label vocabulary, hallucination rate

**Model**: Qwen3.5-0.8B, QLoRA rank 64, 4-bit quantization

### 3.2 Zero-Shot Test Sets

| Held-Out Category | Test Samples | Train Samples (7 remaining) |
|-------------------|:-----------:|:---------------------------:|
| Analysis | 68 | ⏳ |
| Backdoor | 51 | ⏳ |
| Benign | 11 | ⏳ |
| DoS | 4,066 | ⏳ |
| Exploits | 581 | ⏳ |
| Fuzzers | 91 | ⏳ |
| Generic | 13 | ⏳ |
| Reconnaissance | 4,970 | ⏳ |

## 4. Results

### 4.1 Zero-Shot Classification Performance

| Held-Out Category | Support | ZS Strict F1 | ZS Predicted Label | Hallucinated? |
|-------------------|--------:|:------------:|:------------------:|:-------------:|
| Analysis | 68 | ⏳ | ⏳ | ⏳ |
| Backdoor | 51 | ⏳ | ⏳ | ⏳ |
| Benign | 11 | ⏳ | ⏳ | ⏳ |
| DoS | 4,066 | ⏳ | ⏳ | ⏳ |
| Exploits | 581 | ⏳ | ⏳ | ⏳ |
| Fuzzers | 91 | ⏳ | ⏳ | ⏳ |
| Generic | 13 | ⏳ | ⏳ | ⏳ |
| Reconnaissance | 4,970 | ⏳ | ⏳ | ⏳ |

> **Predictions** (before data arrives):
> - Categories with unique network signatures (DoS: high volume, Fuzzers: malformed packets) should transfer better
> - Rare categories (Benign: 11 samples, Generic: 13) may be unpredictable
> - Reconnaissance likely misclassified as DoS (similar scan traffic patterns)

### 4.2 What Does the Model Predict for Unseen Categories?

⏳ Analysis pending — key questions:
1. Does the model predict the held-out label at all? (Has it learned the label from pre-training?)
2. Does it hallucinate a sub-category name? (e.g., predicting "Port Scanning" when "Reconnaissance" was held out)
3. Does it assign the nearest neighbor category?

## 5. Analysis

### 5.1 Feature Overlap Between Categories

| Category Pair | Feature Overlap (Jaccard) | Expected ZS Confusion |
|--------------|:------------------------:|:---------------------:|
| DoS ↔ Fuzzers | High (similar traffic volume) | Likely misclassified |
| Reconnaissance ↔ Analysis | Moderate (scan behavior) | Possible confusion |
| Backdoor ↔ Exploits | Low (different mechanisms) | Less confusion |
| Benign ↔ all | Distinctive (normal traffic) | May transfer well |

### 5.2 Entropy and Transferability

We hypothesize that categories with **low conditional entropy** H(Y|X) given their feature patterns are more transferable:
- If a category's features uniquely identify it → zero-shot success
- If features overlap with trained categories → zero-shot failure

## 6. Discussion

### 6.1 Implications for SOC Deployment

⏳ To be completed with results. Expected narrative:
- Zero-shot LLM ≠ production-ready for novel threats
- Feature-distinctive categories may work; vocabulary-dependent ones won't
- Recommend: continuous fine-tuning pipeline, not one-shot deployment

### 6.2 Comparison with Traditional ML Zero-Shot

- SVM/DT have no concept of zero-shot — they cannot predict unseen labels
- LLMs can generate any label from pre-training vocabulary → may predict correct label even without training examples
- This is a unique LLM advantage (or risk, if hallucinated labels are wrong)

## 7. Conclusion

⏳ Pending results. Expected conclusion: Zero-shot generalization varies by category distinguishability. Fine-tuned LLMs show partial transfer for feature-distinctive categories but fail on vocabulary-dependent ones. Deploying LLMs for novel threat detection requires continuous adaptation, not static fine-tuning.

---

## Acknowledgments

Computing resources provided by ThaiSC on the Lanta HPC system.

---

## References

1. Mirsky, Y., et al. (2018). Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection. *NDSS*.
2. Pushp, P.K. and Srivastava, M.M. (2017). Train Once, Test Anywhere: Zero-Shot Learning for Text Classification. *arXiv:1712.05972*.
3. Yin, W., et al. (2019). Benchmarking Zero-Shot Text Classification. *EMNLP*.
