# Thai-Language SOC Alerts: Cross-Lingual Transfer and Label Hallucination in Non-English Cybersecurity Classification

**Authors**: [Author Names]

---

## Abstract

Cybersecurity LLM research overwhelmingly focuses on English. We investigate whether fine-tuned LLMs can classify SOC alerts presented in Thai, testing cross-lingual transfer and Thai-specific label hallucination patterns. Using SALAD translated to Thai with professional review, we train Qwen3.5-0.8B on Thai-only data and evaluate both Thai→Thai and Thai→English transfer. Our key questions: (1) Does Thai training preserve label compliance (strict F1)? (2) Do models hallucinate Thai translations of MITRE ATT&CK terms? (3) Can a Thai-trained model be deployed in multilingual SOC environments? [⏳ Results pending eval.]

---

## 1. Introduction

SOCs in non-English-speaking countries (Thailand, Japan, Korea) receive alerts in mixed languages. A Thai SOC analyst needs models that understand both English network data and Thai context. We test whether LLM fine-tuning transfers across the language barrier.

### 1.1 Unique Challenges

- MITRE ATT&CK taxonomy is English-only → Thai models must map to English labels
- Thai tokenization differs from English → may affect fine-tuning efficiency
- Thai cybersecurity terminology is unstandardized

## 2. Related Work

### 2.1 Multilingual NLP for Security
Cybersecurity NLP is English-dominated. Rietzke et al. (2023) showed cross-lingual transfer for phishing detection. No prior work evaluates fine-tuned LLMs on Thai cybersecurity classification.

### 2.2 Cross-Lingual Transfer in LLMs
Conneau et al. (2020, XLM-R) demonstrated zero-shot cross-lingual transfer. Multilingual LLMs (Qwen, Llama) have varied Thai support. We test whether fine-tuning on Thai preserves English label compliance.

## 3. Dataset

**SALAD-Thai**: Thai translation of SALAD dataset
- Input: Thai-translated alert descriptions (professional human review)
- Labels: English SALAD labels (same schema)
- Train: 5K samples (Thai input → English labels)
- Test: Standard SALAD test set (English)

## 4. Experiments

| Experiment | Training Data | Test Data | Metric |
|-----------|:-------------|:---------|:-------|
| Thai→Thai | Thai 5K | Thai test | Strict F1 |
| Thai→English | Thai 5K | English SALAD test | Strict F1 |
| English→English | English 5K | English SALAD test | Strict F1 (baseline) |

## 5. Expected Results

### 5.1 Thai Training Performance

| Model | Train Lang | Test Lang | Strict F1 | Halluc |
|-------|:----------:|:---------:|:---------:|:------:|
| Qwen3.5-0.8B | English | English | 0.778 | 1 |
| Qwen3.5-0.8B | Thai | English | ⏳ | ⏳ |
| Qwen3.5-0.8B | Thai | Thai | ⏳ | ⏳ |

### 5.2 Thai-Specific Hallucination Patterns

Expected hallucination types:
1. Thai translations of MITRE terms (e.g., "การสแกนพอร์ต" for "Port Scanning")
2. Romanized Thai (e.g., "kan scan port") 
3. Mixed Thai-English labels

## 6. Discussion

### 6.1 Implications for Multilingual SOC

- If Thai→English transfer works: one model serves multilingual SOCs
- If it fails: need language-specific models per SOC region
- Label standardization across languages becomes critical

## 7. Conclusion

⏳ Pending evaluation results. This study provides the first investigation of non-English LLM fine-tuning for cybersecurity classification.

---

## References

1. Conneau, A., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *ACL*.
2. Rietzke, E., et al. (2023). Cross-Lingual Phishing Detection with Multilingual Transformers. *ACSAC*.
