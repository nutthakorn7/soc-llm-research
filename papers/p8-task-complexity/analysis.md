# P8: When Do LLMs Outperform Traditional ML? An Entropy-Based Decision Framework

## Reframed Title
**"Task Entropy Predicts LLM Necessity: A Decision Framework for Classifier Selection in Applied Domains"**

## Thesis (STRONGEST — validated)
H(Y) predicts: (1) ML-LLM gap, (2) N_sufficient, (3) hallucination rate, (4) strict vs normalized gap

## Key Table

| Domain | H(Y) | DT | SVM | LLM Strict | LLM Norm | Contribution |
|--------|------|-----|------|:----------:|:--------:|-------------|
| SALAD | 1.24 | 87% | 91% | 55-100% | 100% | Semantic ✅, strict varies |
| AG News | 2.00 | 58% | 88% | ⏳ | ⏳ | SVM still OK? |
| GoEmotions | 3.75 | 17% | 24% | ⏳ | ⏳ | **LLM essential** |
| LedGAR | 6.16 | 18% | 65% | ⏳ | ⏳ | **LLM essential** |

## Status
- [x] DT/SVM/BERT baselines ✅
- [x] SALAD strict audit ✅
- [/] P20 evals running (will fill ⏳)
- [ ] Plot + write draft

## Target: ACL Findings / ECML-PKDD
