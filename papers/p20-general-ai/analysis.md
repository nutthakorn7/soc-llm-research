# P20: When, How, and How Big — Cross-Domain Results

## Exact Entropy Values (Computed)

| Domain | Dataset | K | H(Y) bits | H_max | Normalized |
|---|---|---|---|---|---|
| SOC Alerts | SALAD | 13 | **1.2437** | 3.70 | 0.336 |
| News | AG News | 4 | **1.9992** | 2.00 | 1.000 |
| Emotion | GoEmotions | 28 | **3.7453** | 4.81 | 0.779 |
| Legal | LEDGAR | 100 | **6.1580** | 6.64 | 0.927 |

## Traditional ML Results (Done ✅)

| Domain | H(Y) | DT | SVM | LR |
|---|---|---|---|---|
| SALAD | 1.24 | 73.6% | **90.9%** | 72.5% |
| AG News | 2.00 | 57.9% | **88.4%** | 87.4% |
| GoEmotions | 3.75 | 16.9% | **23.8%** | 12.3% |
| LEDGAR | 6.16 | 18.0% | **65.0%** | 53.5% |

## LLM Fine-tuning (0.8B + 9B)

| Domain | 0.8B Training | 9B Training | 0.8B Eval | 9B Eval |
|---|---|---|---|---|
| SALAD | ✅ | ✅ | ✅ | ✅ |
| AG News | ✅ | ✅ | ⏳ | ⏳ |
| GoEmotions | ✅ | ✅ | ⏳ | ⏳ |
| LEDGAR | ✅ | ✅ | ⏳ | ⏳ |

## Entropy vs Best-ML-F1 (Key Figure Data)

| H(Y) | Best Traditional | Best LLM | Gap |
|---|---|---|---|
| 1.24 | SVM 90.9% | LoRA 100% | +9.1% |
| 2.00 | SVM 88.4% | ⏳ | ⏳ |
| 3.75 | SVM 23.8% | ⏳ | ⏳ |
| 6.16 | SVM 65.0% | ⏳ | ⏳ |

## Decision Flowchart (Paper Figure)

```
      Compute H(Y) from labeled data
                  │
          ┌───────┴───────┐
          │               │
      H < 2.0         H ≥ 2.0
          │               │
     SVM/DT ok      ┌─────┴─────┐
    (F1 > 88%)      │           │
                 K ≤ 10      K > 10
                 H < 4       H ≥ 4
                    │           │
                Use ICL    Use LoRA
              (if API ok)  (0.8B enough)
```
