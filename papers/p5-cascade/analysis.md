# P5: Cascade DT→LLM for Cost-Efficient SOC Alert Triage

## Core Idea
Use Decision Tree as **first-pass filter**: only send high-entropy alerts to LLM.

## Architecture
```
Alert → DT Classifier
         ├─ High confidence (>0.95) → DT label (free)
         └─ Low confidence (<0.95) → LLM fine-tuned model ($)
```

## Expected Savings (from P3 + P7 data)

| Task | DT F1 | DT Coverage* | Alerts to LLM | Cost Reduction |
|---|---|---|---|---|
| Classification | 100% | 100% | 0% | **100% saved** |
| Triage | 100% | 100% | 0% | **100% saved** |
| Attack Category | 73.6% | ~70% | ~30% | **70% saved** |

*Coverage = % of alerts where DT confidence > 0.95

## Key Experiment (TODO)
1. Train DT on SALAD → get confidence scores per sample
2. Set threshold sweep (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
3. Route low-confidence to QLoRA-0.8B
4. Measure: overall F1 vs % routed to LLM

## Expected Figure
```
F1 ↑
100% │──────────────────●  (100% LLM)
     │           ●────/
     │        ●─/
 90% │     ●─/
     │  SVM
 74% │●                     (DT only)
     └──────────────────── % sent to LLM →
      0%   10%  20%  50%  100%
```

## Cost Model
- DT inference: ~0.001ms/sample ($0)
- LLM inference: ~100ms/sample ($0.0001)
- Cascade: DT cost + (1-coverage) × LLM cost

## Novelty
First paper to propose **entropy-aware cascade** for SOC:
- If H(task) < 1 bit → DT only (skip LLM)
- If H(task) > 1 bit → cascade DT→LLM
