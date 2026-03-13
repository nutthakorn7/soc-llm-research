# Does Alignment Reduce Hallucination? Comparing SFT, DPO, and ORPO for Label Compliance in SOC Classification

**Authors**: [Author Names]

---

## Abstract

Alignment techniques like Direct Preference Optimization (DPO) and Odds Ratio Preference Optimization (ORPO) are designed to make LLMs follow instructions more precisely. We investigate whether alignment reduces label hallucination in SOC alert classification, where fine-tuned models achieve perfect semantic accuracy but hallucinate MITRE ATT&CK sub-category names. Our pipeline generates preference pairs where correct label strings are "chosen" and hallucinated variants are "rejected," then applies DPO and ORPO to a Qwen3.5-0.8B SFT baseline. We compare strict F1, hallucination rate, and priority score calibration across all three methods. [⏳ Results pending — DPO training running, ORPO eval submitted.]

**Keywords**: DPO, ORPO, alignment, label hallucination, SOC classification

---

## 1. Introduction

Standard SFT (Supervised Fine-Tuning) teaches models *what* to predict but not *how* to express predictions. Alignment techniques add a preference signal: "this output format is preferred over that one." We hypothesize alignment directly addresses label hallucination by teaching the model to prefer exact schema labels over semantically correct but off-schema alternatives.

### 1.1 The Problem

| Method | Semantic Understanding | Label Compliance |
|--------|:---------------------:|:----------------:|
| SFT | ✅ Perfect | ⚠️ Variable (55.7-100%) |
| DPO | ✅ Perfect? | ⏳ Improved? |
| ORPO | ✅ Perfect? | ⏳ Improved? |

## 2. Related Work

### 2.1 Alignment Techniques
RLHF (Ouyang et al., 2022) trains reward models from human preferences. DPO (Rafailov et al., 2023) eliminates the reward model by directly optimizing on preference pairs. ORPO (Hong et al., 2024) unifies SFT and alignment into a single training step. All were designed for safety/helpfulness alignment, not label compliance.

### 2.2 Label Hallucination
Our companion work (P3) identified label vocabulary hallucination as a distinct phenomenon from factual hallucination. We test whether alignment — designed for format compliance — transfers to label compliance.

## 3. Methodology

### 3.1 Preference Data Generation

From SFT model predictions, we automatically construct preference pairs:

```python
# For each SFT prediction where label != strict_match:
chosen = {
    "input": alert_text,
    "output": "Attack Category: Reconnaissance"  # correct schema label
}
rejected = {
    "input": alert_text, 
    "output": "Attack Category: Port Scanning"   # hallucinated label
}
```

**Total pairs**: 5,000 (from SFT mismatches on training set)

### 3.2 Training Pipeline

```
Base model: Qwen3.5-0.8B
    │
    ├── SFT (5K samples) ──► SFT model (baseline)
    │
    ├── SFT + DPO (5K pref pairs) ──► DPO model
    │
    └── SFT + ORPO (5K pref pairs) ──► ORPO model
```

### 3.3 Evaluation

- Strict F1 (Attack Category, exact match)
- Normalized F1 (with alias mapping)
- Hallucinated label count
- Priority Score MAE (calibration)

## 4. Results

### 4.1 Main Comparison

| Method | Strict F1 | Norm F1 | Halluc Labels | PS MAE |
|--------|:---------:|:-------:|:-------------:|:------:|
| SFT | 0.778 | 1.000 | 1 | 0.012 |
| DPO | ⏳ | ⏳ | ⏳ | ⏳ |
| ORPO | ⏳ | ⏳ | ⏳ | ⏳ |

### 4.2 Hallucination Reduction

| Method | "Port Scanning" count | "Backdoors" count | Total off-schema |
|--------|:---------------------:|:-----------------:|:----------------:|
| SFT | 4,895 | 20 | 4,921 |
| DPO | ⏳ | ⏳ | ⏳ |
| ORPO | ⏳ | ⏳ | ⏳ |

### 4.3 Priority Score Calibration

| Method | MAE | Calibration Curve |
|--------|:---:|:-----------------:|
| SFT | 0.012 | ⏳ |
| DPO | ⏳ | ⏳ |
| ORPO | ⏳ | ⏳ |

## 5. Analysis

### 5.1 Expected Outcomes

**If alignment helps**: DPO/ORPO should reduce "Port Scanning" predictions to 0, achieving strict F1 close to 100%. This would prove that alignment is a cheaper alternative to increasing training data from 5K to 20K.

**If alignment doesn't help**: The hallucination is too deeply embedded in pre-training weights for preference optimization to override. This would suggest that label hallucination is a fundamentally different problem from alignment failures.

### 5.2 Cost Comparison

| Method | Training Cost | Extra Data Needed | Strict F1 |
|--------|:------------:|:-----------------:|:---------:|
| SFT (5K) | $2.24 | None | 0.778 |
| SFT (20K) | $4.48 | +15K labels | 1.000 |
| DPO (5K + 5K pref) | ~$3.50 | +5K auto-generated | ⏳ |
| ORPO (5K + 5K pref) | ~$3.00 | +5K auto-generated | ⏳ |
| Reasoning model (5K) | $0.60 | None | 1.000 |

> If DPO achieves >95% strict F1, it's cheaper than 20K SFT but more expensive than using a reasoning model.

## 6. Conclusion

⏳ Pending results. Expected contribution: first study of alignment techniques for reducing label hallucination (as opposed to safety alignment). If successful, provides a new tool for improving label compliance without additional labeled data.

---

## References

1. Hong, J., et al. (2024). ORPO: Monolithic Preference Optimization without Reference Model. *arXiv:2403.07691*.
2. Ouyang, L., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS*.
3. Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. *NeurIPS*.
