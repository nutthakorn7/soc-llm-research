# P5: Cascade DT→LLM for Cost-Efficient SOC Alert Triage

## Core Idea
Use Decision Tree as **first-pass filter**: only send high-entropy alerts to LLM.

## Architecture
```
Alert → DT Classifier
         ├─ High confidence (>0.95) → DT label (free)
         └─ Low confidence (<0.95) → LLM fine-tuned model ($)
```

## Actual Cascade Results (SALAD)

| Threshold | LR handles | LLM handles | Cascade F1 | Cost Savings |
|---|---|---|---|---|
| 0.50 | 99.9% | 0.1% | 100.0% | 99.9% |
| 0.80 | 99.7% | 0.3% | 100.0% | 99.7% |
| 0.90 | 98.9% | 1.1% | 100.0% | 98.9% |
| 0.95 | 91.8% | 8.2% | 100.0% | 91.8% |

### Key Finding
> **SALAD is too easy for cascade benefit!** LR confidence is >99.7% for almost all samples.
> Cascade shines on **high-entropy datasets** (GoEmotions H=3.75, LEDGAR H=6.16).
> → Must run cascade experiment on cross-domain data to show value.

## Cross-Domain Cascade (TODO — need P20 eval results)

| Domain | H(Y) | Traditional ML | Expected LLM% routed |
|---|---|---|---|
| SALAD | 1.24 | 90.9% | ~0% (cascade useless) |
| AG News | 2.00 | 88.4% | ~5-10% |
| GoEmotions | 3.75 | 23.8% | ~60-80% |
| LEDGAR | 6.16 | 65.0% | ~30-50% |

## Novelty
First paper to propose **entropy-aware cascade** for SOC:
- If H(task) < 1 bit → DT only (skip LLM)
- If H(task) > 1 bit → cascade DT→LLM
