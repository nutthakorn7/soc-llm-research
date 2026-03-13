# Beyond SALAD: A Comparative Study of Cybersecurity NLP Datasets for LLM Evaluation

**Authors**: [Author Names]

---

## Abstract

The rapid growth of LLM-based cybersecurity research has outpaced the development of appropriate evaluation benchmarks. We systematically analyze 6 cybersecurity NLP datasets across 12 dimensions — including label entropy, unique pattern count, class ambiguity, and LLM suitability — revealing that most existing benchmarks are too easy for modern LLMs (H(Y) < 2 bits). SALAD achieves 100% normalized F1 with a Decision Tree and has only 87 unique patterns. We propose a difficulty grading system (Grades A-D) and identify characteristics needed for challenging LLM benchmarks: H(Y) > 3 bits, >5K unique patterns, multi-label output, and adversarial examples. This analysis guides the cybersecurity NLP community toward creating benchmarks that can meaningfully differentiate between LLM architectures.

---

## 1. Introduction

Cybersecurity NLP has produced multiple datasets, but no systematic comparison exists. Researchers often select benchmarks based on availability rather than suitability, leading to inflated performance claims. Our analysis reveals that many popular cybersecurity datasets are trivially solvable by simple classifiers, providing no signal about LLM capability.

## 2. Related Work

### 2.1 Cybersecurity Benchmark Surveys
Ferrag et al. (2024) surveyed LLMs for cybersecurity but focused on model capabilities, not dataset difficulty. Catal et al. (2022) reviewed network intrusion datasets but without entropy-based analysis. No prior work grades cybersecurity datasets by LLM evaluation suitability.

### 2.2 Dataset Difficulty Analysis
Swayamdipta et al. (2020) proposed dataset cartography to map training dynamics. Ethayarajh et al. (2022) discussed utility vs. benchmark gaming. We adapt these concepts to cybersecurity, introducing entropy-based difficulty grading.

### 2.3 Benchmark Design
DynaBench (Kiela et al., 2021) advocates human-in-the-loop benchmark creation. GLUE/SuperGLUE (Wang et al., 2019) established multi-task NLU benchmarks. We argue cybersecurity needs equivalent rigor in benchmark design.

## 3. Datasets Analyzed

| Dataset | Size | Type | K | H(Y) | Year |
|---------|:----:|------|:-:|:----:|:----:|
| **SALAD** (ours) | 1.9M | Classification | 8 | 1.24 | 2025 |
| **Advanced SIEM** | 100K | Classification | 6 | 0.85 | 2024 |
| **Trendyol Cyber** | 53K | Instruction | — | — | 2024 |
| **Fenrir v2.0** | 84K | Threat Intel | — | — | 2024 |
| **soc-audit-11k** | 11.5K | Q&A | — | — | 2024 |
| **BBC News** | 1.2K | Classification | 5 | 2.32 | 2017 |

## 4. 12-Dimension Comparison

### 4.1 Statistical Properties

| Dataset | Samples | Labels | H(Y) | Max Class % | Unique Patterns |
|---------|--------:|:------:|:----:|:-----------:|:---------------:|
| SALAD | 1.9M | 8 | 1.24 | 99.9% | 87 |
| SIEM | 100K | 6 | 0.85 | 82.1% | ⏳ |
| Trendyol | 53K | — | — | — | ⏳ |
| Fenrir | 84K | — | — | — | ⏳ |
| soc-audit | 11.5K | — | — | — | ⏳ |
| BBC News | 1.2K | 5 | 2.32 | 23.7% | ⏳ |

### 4.2 LLM Suitability Scoring

| Criterion | SALAD | SIEM | BBC |
|-----------|:-----:|:----:|:---:|
| H(Y) > 2 bits | ❌ 1.24 | ❌ 0.85 | ✅ 2.32 |
| >1K unique patterns | ❌ 87 | ⏳ | ⏳ |
| Class balance (max <50%) | ❌ 99.9% | ❌ 82.1% | ✅ 23.7% |
| Ambiguous patterns | ❌ 0% | ⏳ | ⏳ |
| DT F1 < 70% | ❌ 87.4% | ⏳ | ✅ 58.1% |
| **Difficulty Grade** | **D (trivial)** | **D (trivial)** | **C (moderate)** |

### 4.3 Difficulty Grading System

| Grade | H(Y) | DT F1 | Unique Patterns | Example |
|:-----:|:----:|:-----:|:---------------:|---------|
| **A** (challenging) | >4 bits | <30% | >10K | — |
| **B** (moderate-hard) | 3-4 bits | 30-60% | >5K | GoEmotions? |
| **C** (moderate) | 2-3 bits | 60-80% | >1K | BBC, AG News |
| **D** (trivial) | <2 bits | >80% | <1K | SALAD, SIEM |

## 5. What Makes a Good LLM Benchmark?

### 5.1 Requirements

1. **H(Y) > 3 bits**: Enough label diversity to require genuine understanding
2. **>5K unique patterns**: Prevents lookup-table solutions
3. **Ambiguity > 5%**: Some patterns should map to multiple valid labels
4. **Max class < 30%**: Balanced enough for macro-F1 to be meaningful
5. **DT baseline < 50%**: Simple classifiers should struggle

### 5.2 Gap Analysis

No existing cybersecurity dataset meets all 5 criteria. The community urgently needs:
- Higher-entropy classification tasks (100+ cyber categories)
- Multi-label threat classification (one alert, multiple TTPs)
- Open-ended generation tasks (incident report writing)

## 6. Discussion

### 6.1 Why Most Cybersecurity Datasets Are Grade D
Cybersecurity datasets are typically created by security engineers, not NLP researchers. They focus on coverage (many alerts) rather than diversity (many distinct patterns). SALAD's 1.9M samples contain only 87 unique patterns — a lookup table would suffice.

### 6.2 Toward Grade A Cybersecurity Benchmarks
A Grade A dataset would require:
- Real SOC analyst disagreement data (inherent ambiguity)
- Multi-TTP labeling (one alert → multiple MITRE techniques)
- Temporal concept drift (attack patterns change over time)
- Adversarial examples (alerts designed to evade classification)

## 7. Conclusion

Our 12-dimension analysis reveals that existing cybersecurity NLP benchmarks are insufficient for evaluating modern LLMs. SALAD (H=1.24, 87 patterns) and SIEM (H=0.85) are Grade D — trivially solvable by Decision Trees. We propose a difficulty grading system and identify 5 requirements for challenging cybersecurity LLM benchmarks, calling on the community to develop Grade A-B datasets.

---

## References

1. Catal, C., et al. (2022). A Survey of Network Intrusion Detection Datasets. *Computers & Security*.
2. Ethayarajh, K., et al. (2022). Utility is in the Eye of the User. *NeurIPS*.
3. Ferrag, M.A., et al. (2024). LLMs for Cyber Security: A Survey. *arXiv:2405.12750*.
4. Kiela, D., et al. (2021). Dynabench: Rethinking Benchmarking in NLP. *NAACL*.
5. Swayamdipta, S., et al. (2020). Dataset Cartography: Mapping Training Dynamics. *EMNLP*.
6. Wang, A., et al. (2019). SuperGLUE: Stickier Benchmark for NLU. *NeurIPS*.

---

## Acknowledgments

Computing resources provided by ThaiSC on the Lanta HPC system.
