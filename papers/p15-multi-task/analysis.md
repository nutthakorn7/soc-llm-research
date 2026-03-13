# P15: Multi-Task vs Single-Task Learning for SOC Alerts

## Experiment Design

| Config | Tasks | Seeds | Training | Eval |
|--------|-------|-------|----------|------|
| Multi-task | All 3 | 3 | ✅ (P3 data) | ✅ 100% F1 |
| Single-cls | Cls only | 3 | ✅ | 🔄 submitted |
| Single-tri | Tri only | 3 | ✅ | 🔄 submitted |
| Single-atk | Atk only | 3 | ✅ | 🔄 submitted |

**9 single-task evals submitted** on Lanta.

## ⚠️ Expected Challenge
Multi-task = 100% F1. Single-task likely also 100% on its own task (SALAD too easy).

Differentiation must come from:
1. Multi-task advantage on **rare classes** (Backdoor: 51 samples, Generic: 13)
2. Inference efficiency (single-task = shorter output = faster)
3. Cross-domain comparison (higher entropy tasks)

## Action Plan
- [x] All 9 single-task trainings ✅
- [/] 9 eval jobs submitted
- [ ] Compare multi vs single per-class F1
- [ ] Write paper
