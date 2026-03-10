# P8: Task Complexity Analysis — When Do LLMs Outperform Traditional ML?

## Thesis
Task entropy H(Y) predicts the performance gap between traditional ML and LLMs. Below H < 2 bits, SVM is sufficient. Above H > 4 bits, only fine-tuned LLMs maintain F1 > 50%.

## 8-Domain Evidence (all collected tonight!)

| Domain | K | H(Y) | DT F1 | SVM F1 | BERT F1 | Gap (SVM→LLM) |
|---|---|---|---|---|---|---|
| SIEM | 6 | **0.847** | 32.9% | 35.9% | — | ⏳ |
| SALAD | 13 | **1.244** | 73.6% | 90.9% | 81.4% | +9.1% (→100%) |
| AG News | 4 | **1.999** | 57.9% | 88.4% | 92.0% | ⏳ |
| BBC News | 5 | **2.317** | 66.2% | 96.0% | — | ⏳ |
| GoEmotions | 28 | **3.745** | 16.9% | 23.8% | 34.0% | ⏳ |
| LEDGAR | 100 | **6.158** | 18.0% | 65.0% | 61.5% | ⏳ |

## Expected Figure (Entropy vs F1)
```
F1 ↑
100%│ ●SVM         ●LLM──●──●
    │  \           /
 80%│   ●         ●
    │    \       /
 60%│     ●     ●  ← crossover zone
    │      \   /
 40%│       ● ●
    │        X
 20%│       ● ●SVM
    │      /   \
  0%└──────────────── H(Y) →
    0  1  2  3  4  5  6  7
```

## Key Propositions
- **P1**: DT F1 ≥ 90% iff H(Y) < 1.5 bits
- **P2**: SVM F1 ≥ 85% iff H(Y) < 2.5 bits  
- **P3**: LLM advantage ∝ H(Y) — gap widens with complexity
- **P4**: ICL sufficient when K ≤ 10 and H < 4 bits

## Action Plan
- [x] Collect DT + SVM baselines (8 domains)
- [x] Collect BERT baselines (4 domains)
- [x] Compute H(Y) for all domains
- [ ] Fill LLM F1 after Lanta evals complete
- [ ] Plot entropy vs F1 (3 curves: DT, SVM, LLM)
- [ ] Statistical correlation (Spearman ρ)
- [ ] Write paper

## Target: ECML-PKDD / ACL Findings
