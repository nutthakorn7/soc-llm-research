# P5: Cascade DT→LLM for Cost-Efficient SOC Alert Triage

## Core Idea
Use Decision Tree as **first-pass filter**: only send uncertain alerts to LLM.

## ⚠️ Critical Finding: SALAD Too Easy for Cascade
LR confidence > 99.7% for all samples → LLM never triggered.

| Threshold | DT handles | LLM handles | Cascade F1 | Cost Savings |
|-----------|-----------|-------------|------------|-------------|
| 0.50 | 99.9% | 0.1% | 100.0% | 99.9% |
| 0.95 | 91.8% | 8.2% | 100.0% | 91.8% |

## Reframed: Cross-Domain Cascade (needs P20 results)

| Domain | H(Y) | DT F1 | Expected LLM% routed |
|--------|------|-------|---------------------|
| SALAD | 1.24 | 87.4% | ~0% (cascade useless) |
| AG News | 2.00 | 57.9% | ~10-20% |
| GoEmotions | 3.75 | 16.9% | ~60-80% |
| LedGAR | 6.16 | 18.0% | ~30-50% |

## Novelty: Entropy-Aware Cascade
- If H(task) < 1.5 → DT only (skip LLM entirely)
- If H(task) > 1.5 → cascade DT→LLM

## Action Plan
- [x] Cascade results on SALAD (trivial)
- [/] Cross-domain cascade (needs P20 eval results)
- [ ] Write paper
