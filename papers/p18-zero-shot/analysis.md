# P18: Zero-Shot Transfer Across Attack Categories

## Experiment: Leave-One-Category-Out
Train on 7 categories, test on unseen 8th.

## All 16 Models Trained ✅

| Held-out Category | 0.8B | 9B |
|-------------------|------|-----|
| Analysis | ✅ zs-no-analysis | ✅ zs9b-no-analysis |
| Backdoor | ✅ | ✅ |
| Benign | ✅ | ✅ |
| DoS | ✅ | ✅ |
| Exploits | ✅ | ✅ |
| Fuzzers | ✅ | ✅ |
| Generic | ✅ | ✅ |
| Reconnaissance | ✅ | ✅ |

- ✅ `zero_shot_results.json` exists
- 9B eval adapters all synced

## Key Questions
1. Can model identify "Reconnaissance" if never trained on it?
2. Which categories transfer well? (DoS↔Exploits?)
3. 9B > 0.8B on zero-shot?

## ⚠️ Note on SALAD Simplicity
Given only 87 unique prompts, zero-shot might still work by pattern matching on network features. This would be an interesting finding — "even zero-shot transfer is easy when H(Y) is low."

## Action Plan
- [x] 16 models trained ✅
- [x] Results JSON exists ✅
- [ ] Analyze transfer matrix
- [ ] Write paper

## Target: ACL Findings / EMNLP
