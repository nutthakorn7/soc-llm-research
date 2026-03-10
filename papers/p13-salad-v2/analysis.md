# P13: SALAD-v2 — Multi-Source SOC Dataset

## Thesis
A unified, multi-source cybersecurity dataset combining SALAD (SOC alerts) and Advanced SIEM (SIEM logs) enables cross-source generalization evaluation.

## SALAD-v2 Stats
| Metric | SALAD | SIEM | **SALAD-v2** |
|---|---|---|---|
| Train | 5,000 | 5,000 | **10,000** |
| Test | 9,851 | 1,000 | **10,851** |
| Classes | 13 | 6 | **46** |
| H(Y) | 1.244 | 0.847 | **2.114** |

## Cross-Source Evaluation Plan
1. Train SALAD → Test SIEM (zero-shot transfer)
2. Train SIEM → Test SALAD (reverse transfer)
3. Train SALAD-v2 → Test both (multi-source boost?)

## Key Research Questions
- Does multi-source training improve generalization?
- How much does domain shift hurt across SOC/SIEM?
- Can a single model handle both data types?

## Action Plan
- [x] Merge SALAD + SIEM datasets
- [x] Submit training on SALAD-v2 (Lanta)
- [ ] Eval: SALAD-v2 model on SALAD test
- [ ] Eval: SALAD-v2 model on SIEM test
- [ ] Eval: SALAD-only model on SIEM test (transfer)
- [ ] Compare F1 across all configs
- [ ] Write paper

## Future: Add More Sources
- Fenrir v2.0 (84K, threat intel) → SALAD-v3
- Trendyol (53K, instruction) → SALAD-v3
- Unified MITRE ATT&CK label mapping

## Target: NeurIPS Datasets & Benchmarks Track
